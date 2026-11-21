"""
AI counting solution using pretrained Ultralytics FastSAM segmentation.

This does not require custom training. It generates segmentation masks,
filters masks that are likely to be individual physical objects, counts them,
and writes visual overlays.

Usage:
    python ai_counting.py --input <dataset> --output <output>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import FastSAM


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _split_large_mask(
    mask: np.ndarray,
    image: np.ndarray,
    reference_area: float,
) -> list[np.ndarray]:
    area = int(mask.sum())
    if area <= reference_area * 1.5:
        return [mask]

    x1, y1, x2, y2 = _mask_box(mask)
    roi_mask = mask[y1:y2, x1:x2].astype(np.uint8)
    distance = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    radius = max(1.0, float(np.sqrt(reference_area / np.pi)))
    peak_kernel = max(3, int(round(radius * 1.2)) | 1)
    local_maxima = distance >= cv2.dilate(
        distance, np.ones((peak_kernel, peak_kernel), dtype=np.uint8)
    ) - 1e-3
    peak_floor = max(1.0, radius * 0.35)
    peaks = np.uint8(local_maxima & (distance >= peak_floor))
    peak_count, peak_labels = cv2.connectedComponents(peaks)
    if peak_count <= 2:
        return [mask]

    markers = np.zeros(roi_mask.shape, dtype=np.int32)
    markers[roi_mask == 0] = 1
    markers[peak_labels > 0] = peak_labels[peak_labels > 0] + 1
    cv2.watershed(image[y1:y2, x1:x2].copy(), markers)

    minimum_piece_area = max(50, int(reference_area * 0.20))
    pieces = []
    for label in range(2, peak_count + 1):
        piece_roi = (markers == label) & (roi_mask > 0)
        if int(piece_roi.sum()) < minimum_piece_area:
            continue
        piece = np.zeros_like(mask)
        piece[y1:y2, x1:x2] = piece_roi
        pieces.append(piece)

    if len(pieces) < 2:
        return [mask]
    covered = np.logical_or.reduce(pieces)
    if int(covered.sum()) < area * 0.70:
        return [mask]
    return pieces


def filter_masks(
    masks: list[np.ndarray],
    image_shape,
    image: np.ndarray | None = None,
) -> list[np.ndarray]:
    if not masks:
        return []

    h, w = image_shape[:2]
    image_area = h * w
    minimum_area = max(80, int(image_area * 0.00010))
    candidates = []

    for m in masks:
        m = m.astype(bool)
        area = int(m.sum())
        if area < minimum_area:
            continue
        if area > image_area * 0.45:
            continue

        ys, xs = np.where(m)
        if len(xs) == 0:
            continue

        bw = xs.max() - xs.min() + 1
        bh = ys.max() - ys.min() + 1
        fill = area / max(1, bw * bh)
        if fill < 0.02:
            continue

        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        candidates.append((m, area, box))

    candidates.sort(key=lambda item: item[1], reverse=True)
    selected = []
    for m, area, box in candidates:
        duplicate_index = None
        x1, y1, x2, y2 = box
        for index, (selected_mask, selected_area, selected_box) in enumerate(selected):
            sx1, sy1, sx2, sy2 = selected_box
            ix1, iy1 = max(x1, sx1), max(y1, sy1)
            ix2, iy2 = min(x2, sx2), min(y2, sy2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            intersection = np.logical_and(
                m[iy1:iy2, ix1:ix2],
                selected_mask[iy1:iy2, ix1:ix2],
            ).sum()
            union = area + selected_area - intersection
            if union and (
                intersection / union > 0.85
                or intersection / min(area, selected_area) > 0.90
            ):
                duplicate_index = index
                break
        if duplicate_index is not None:
            merged = np.logical_or(selected[duplicate_index][0], m)
            selected[duplicate_index] = (
                merged,
                int(merged.sum()),
                _mask_box(merged),
            )
            continue
        selected.append((m, area, box))

    if not selected:
        return []

    reference_area = float(np.median([item[1] for item in selected]))
    if image is None:
        image = np.zeros((h, w, 3), dtype=np.uint8)
    split_masks = []
    for mask, _, _ in selected:
        split_masks.extend(_split_large_mask(mask, image, reference_area))
    return split_masks


def overlay(image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    out = image.copy()

    for idx, mask in enumerate(masks, start=1):
        fill = np.zeros_like(out)
        # Deterministic pseudo-color, no external palette required.
        fill[:, :, 0] = (37 * idx) % 255
        fill[:, :, 1] = (97 * idx) % 255
        fill[:, :, 2] = (157 * idx) % 255
        out[mask] = cv2.addWeighted(out[mask], 0.55, fill[mask], 0.45, 0)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 2)

        ys, xs = np.where(mask)
        if len(xs):
            x, y = int(xs.min()), int(ys.min())
            cv2.putText(out, str(idx), (x, max(20, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return out


def instance_mask(masks: list[np.ndarray], shape) -> np.ndarray:
    out = np.zeros(shape[:2], dtype=np.uint16)
    for i, m in enumerate(masks, start=1):
        out[m] = i
    return out


def process_one(model: FastSAM, path: Path, output_root: Path) -> dict:
    image = cv2.imread(str(path))
    if image is None:
        return {"image": path.as_posix(), "status": "read_error"}

    original_h, original_w = image.shape[:2]
    scale = min(1.0, 1600 / max(original_h, original_w))
    work = (
        cv2.resize(
            image,
            (int(original_w * scale), int(original_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        if scale != 1.0
        else image
    )

    results = model(work, verbose=False, conf=0.37)
    if not results:
        masks = []
    else:
        r = results[0]
        if r.masks is None:
            masks = []
        else:
            masks = []
            for polygon in r.masks.xy:
                mask = np.zeros(work.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
                masks.append(mask.astype(bool))

    masks = filter_masks(masks, work.shape, work)
    output_mask = instance_mask(masks, work.shape)
    output_overlay = overlay(work, masks)
    if scale != 1.0:
        output_mask = cv2.resize(
            output_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST
        )
        output_overlay = cv2.resize(
            output_overlay, (original_w, original_h), interpolation=cv2.INTER_LINEAR
        )

    out_dir = output_root / path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / f"{path.stem}_mask.png"
    overlay_path = out_dir / f"{path.stem}_overlay.jpg"
    if not cv2.imwrite(str(mask_path), output_mask):
        raise OSError(f"Could not write mask: {mask_path}")
    if not cv2.imwrite(str(overlay_path), output_overlay):
        raise OSError(f"Could not write overlay: {overlay_path}")

    return {
        "image": path.as_posix(),
        "count": len(masks),
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

    merged.to_csv(gt_path.parent / "ai_evaluation.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="FastSAM-s.pt")
    parser.add_argument("--ground-truth", type=Path)
    args = parser.parse_args()

    images = discover_images(args.input)
    if not images:
        raise SystemExit(f"No images found under {args.input}")

    print(f"Loading segmentation model: {args.model}")
    model = FastSAM(args.model)

    results = []
    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path}")
        try:
            results.append(process_one(model, path, args.output))
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({"image": path.as_posix(), "status": "error", "error": str(exc)})

    df = pd.DataFrame(results)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "predictions.csv", index=False)

    if args.ground_truth:
        evaluate(df, args.ground_truth)

    print(f"\nDone. Results written to {args.output}")


if __name__ == "__main__":
    main()
