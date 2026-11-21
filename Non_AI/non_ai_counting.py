"""
Non-AI counting solution for the PerspectivLabs CountingChallenge.

No neural network is used.

The pipeline:
1. Discovers images recursively.
2. Builds several foreground hypotheses.
3. Cleans masks using morphology.
4. Uses connected components and watershed.
5. Filters components using robust geometric statistics.
6. Writes count, binary mask and overlay.
7. Optionally evaluates against ground_truth.csv.

Usage:
    python non_ai_counting.py --input ../data --output ../outputs/non_ai
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 else n + 1


def resize_for_processing(image: np.ndarray, max_side: int = 1600):
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image, 1.0
    out = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return out, scale


def border_foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Estimate whether the object is darker/lighter than the border background."""
    h, w = gray.shape
    b = max(5, min(h, w) // 30)
    border = np.concatenate([
        gray[:b, :].ravel(), gray[-b:, :].ravel(),
        gray[:, :b].ravel(), gray[:, -b:].ravel()
    ])
    bg = float(np.median(border))

    dark = cv2.inRange(gray, 0, max(0, bg - max(10, np.std(border) * 0.8)))
    light = cv2.inRange(gray, min(255, bg + max(10, np.std(border) * 0.8)), 255)

    # Choose the polarity that produces more plausible foreground coverage.
    candidates = [dark, light]
    scored = []
    for m in candidates:
        ratio = np.count_nonzero(m) / m.size
        scored.append(abs(ratio - 0.20))
    return candidates[int(np.argmin(scored))]


def candidate_masks(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    masks = []

    # Otsu both polarities.
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks += [m, cv2.bitwise_not(m)]

    # Adaptive threshold both polarities.
    block = odd(max(31, min(gray.shape) // 12))
    a = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, 5
    )
    masks += [a, cv2.bitwise_not(a)]

    # Border-based segmentation.
    masks.append(border_foreground_mask(gray))

    # HSV saturation/value based candidate.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    _, sm = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, vm = cv2.threshold(val, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks += [sm, cv2.bitwise_not(sm), vm, cv2.bitwise_not(vm)]

    return masks


def clean_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    k = odd(max(3, min(h, w) // 150))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel * 2)
    return m


def watershed_instances(binary: np.ndarray) -> tuple[np.ndarray, int]:
    binary = (binary > 0).astype(np.uint8) * 255
    if cv2.countNonZero(binary) == 0:
        return np.zeros(binary.shape, dtype=np.uint16), 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return np.zeros(binary.shape, dtype=np.uint16), 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    median_area = float(np.median(areas)) if len(areas) else 1.0

    # Start with connected components.
    good = np.zeros_like(binary)
    min_area = max(20, median_area * 0.08)
    max_area = max(min_area * 5, median_area * 8)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            good[labels == i] = 255

    if cv2.countNonZero(good) == 0:
        good = binary

    # Distance transform and watershed to split touching objects.
    dist = cv2.distanceTransform(good, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return connected_component_instances(good)

    # Adaptive foreground peaks.
    threshold = max(0.25 * dist.max(), np.percentile(dist[dist > 0], 65))
    sure_fg = np.uint8(dist >= threshold) * 255

    bg = cv2.dilate(good, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(bg, sure_fg)

    nc, markers = cv2.connectedComponents(sure_fg)
    if nc <= 1:
        return connected_component_instances(good)

    markers = markers + 1
    markers[unknown == 255] = 0

    color = cv2.cvtColor(good, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)

    instance_mask = np.zeros(good.shape, dtype=np.uint16)
    instance_count = 0
    for idx in range(2, markers.max() + 1):
        region = markers == idx
        area = int(region.sum())
        if area >= min_area * 0.35:
            instance_count += 1
            instance_mask[region] = instance_count

    return instance_mask, instance_count


def connected_component_instances(binary: np.ndarray) -> tuple[np.ndarray, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), 8
    )
    instances = np.zeros(binary.shape, dtype=np.uint16)
    instance_count = 0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] == 0:
            continue
        instance_count += 1
        instances[labels == label] = instance_count
    return instances, instance_count


def score_candidate(mask: np.ndarray) -> tuple[float, int, np.ndarray]:
    cleaned = clean_mask(mask)
    instance_mask, count = watershed_instances(cleaned)

    if count == 0:
        return 1e9, 0, instance_mask

    h, w = mask.shape
    image_area = h * w
    mean_instance_area = np.count_nonzero(instance_mask) / count
    count_limit = max(250, int(math.sqrt(image_area) * 0.20))
    minimum_mean_area = max(40, image_area * 0.00003)
    if count > count_limit and mean_instance_area < minimum_mean_area:
        return math.inf, count, instance_mask

    ratio = np.count_nonzero(instance_mask) / float(h * w)

    # Keep the count unweighted until more verified labels are available.
    coverage_penalty = abs(math.log(max(ratio, 1e-5) / 0.06))
    count_penalty = 0.0

    binary_instances = (instance_mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_instances, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas):
        log_areas = np.log(np.maximum(areas, 1))
        dispersion = float(np.std(log_areas))
    else:
        dispersion = 10.0

    # Lower score is better.
    print("Count:", count, "Coverage:", round(coverage_penalty, 3), "Dispersion:", round(dispersion, 3))
    score = coverage_penalty + 0.15 * dispersion + count_penalty
    return score, count, instance_mask


def overlay(image: np.ndarray, instance_mask: np.ndarray) -> np.ndarray:
    out = image.copy()

    fill = np.zeros_like(out)
    fill[:, :, 1] = np.where(instance_mask > 0, 255, 0).astype(np.uint8)
    out = cv2.addWeighted(out, 0.70, fill, 0.30, 0)

    for instance_id in range(1, int(instance_mask.max()) + 1):
        region = (instance_mask == instance_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(out, [contour], -1, (0, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(out, str(instance_id), (x, max(20, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return out


def process_one(path: Path, output_root: Path) -> dict:
    image = cv2.imread(str(path))
    if image is None:
        return {"image": path.as_posix(), "status": "read_error"}

    original_h, original_w = image.shape[:2]
    work, scale = resize_for_processing(image)

    best = None
    for candidate in candidate_masks(work):
        score, count, inst = score_candidate(candidate)
        ratio = np.count_nonzero(inst) / float(inst.shape[0] * inst.shape[1])
        print("Candidate count:", count, "Score:", score, "Coverage:", round(ratio, 4))
        if best is None or score < best[0]:
            best = (score, count, inst)

    _, count, instance_mask = best

    if scale != 1.0:
        instance_mask = cv2.resize(instance_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    rel = path.name
    out_dir = output_root / path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / f"{path.stem}_mask.png"), instance_mask)
    cv2.imwrite(str(out_dir / f"{path.stem}_overlay.jpg"), overlay(image, instance_mask))

    return {
        "image": path.as_posix(),
        "count": int(count),
        "status": "ok"
    }


def evaluate(results: pd.DataFrame, gt_path: Path):
    gt = pd.read_csv(gt_path)
    if not {"image", "count"}.issubset(gt.columns):
        raise ValueError("ground_truth.csv must contain columns: image,count")

    gt["image_key"] = gt["image"].astype(str).map(lambda x: Path(x).name)
    results["image_key"] = results["image"].astype(str).map(lambda x: Path(x).name)
    merged = gt.merge(results, on="image_key", suffixes=("_gt", "_pred"))

    if merged.empty:
        print("No ground-truth rows matched prediction filenames.")
        return

    merged["abs_error"] = (merged["count_gt"] - merged["count_pred"]).abs()
    merged["exact"] = merged["abs_error"] == 0
    merged["count_accuracy"] = 1 - merged["abs_error"] / merged["count_gt"].clip(lower=1)

    print("\nEvaluation")
    print("----------")
    print(f"Images evaluated : {len(merged)}")
    print(f"Exact count      : {merged['exact'].mean() * 100:.2f}%")
    print(f"Mean count acc.  : {merged['count_accuracy'].mean() * 100:.2f}%")
    print(f"MAE              : {merged['abs_error'].mean():.4f}")

    merged.to_csv(gt_path.parent / "non_ai_evaluation.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ground-truth", type=Path)
    args = parser.parse_args()

    images = discover_images(args.input)
    if not images:
        raise SystemExit(f"No images found under {args.input}")

    results = []
    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path}")
        results.append(process_one(path, args.output))

    df = pd.DataFrame(results)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "predictions.csv", index=False)

    if args.ground_truth:
        evaluate(df, args.ground_truth)

    print(f"\nDone. Results written to {args.output}")


if __name__ == "__main__":
    main()
