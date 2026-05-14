"""
Non-AI Object Counting using OpenCV
=====================================
Counts screws/bolts in images and overlays segmentation masks.

Techniques:
  - Adaptive preprocessing with multiple binarisation strategies
  - Morphological operations (opening, closing, dilation)
  - Watershed algorithm for touching/overlapping objects
  - Contour-based fallback for well-separated objects
  - Colour-coded mask overlay with per-object labels
  - Auto-selects best strategy based on image characteristics
"""

import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path


# ──────────────────────────────────────────────
# Preprocessing strategies
# ──────────────────────────────────────────────

def preprocess_otsu(image: np.ndarray) -> np.ndarray:
    """Otsu global threshold — works well on uniform backgrounds."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def preprocess_adaptive(image: np.ndarray) -> np.ndarray:
    """Adaptive threshold — handles uneven lighting well."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=8
    )
    return binary


def preprocess_canny_filled(image: np.ndarray) -> np.ndarray:
    """Canny edges + filled contours — great for metallic objects."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    # Fill contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(gray)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return filled


def clean_binary(binary: np.ndarray,
                 kernel_size: int = 5,
                 iterations: int = 2) -> np.ndarray:
    """Remove noise and fill holes with morphological ops."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


# ──────────────────────────────────────────────
# Segmentation — Watershed
# ──────────────────────────────────────────────

def watershed_segment(image: np.ndarray,
                      binary: np.ndarray,
                      fg_thresh: float = 0.4) -> tuple:
    """
    Watershed segmentation to split touching objects.
    Returns (markers, count).
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist, fg_thresh * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)

    unique = np.unique(markers)
    count = int(np.sum(unique > 1))
    return markers, count


# ──────────────────────────────────────────────
# Segmentation — Contour-based
# ──────────────────────────────────────────────

def contour_count(binary: np.ndarray,
                  min_area: int = 200,
                  max_area: int = None) -> tuple:
    """Contour counting for well-separated objects."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary.shape
    if max_area is None:
        max_area = (h * w) // 4   # no single object should be > 25% of image

    valid = [c for c in contours
             if min_area <= cv2.contourArea(c) <= max_area]
    return valid, len(valid)


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────

def draw_watershed_masks(image: np.ndarray,
                          markers: np.ndarray) -> np.ndarray:
    """Colour-coded watershed mask overlay."""
    overlay = image.copy()
    output  = image.copy()

    unique_labels = np.unique(markers)
    object_labels = unique_labels[unique_labels > 1]

    rng = np.random.default_rng(42)
    colours = rng.integers(60, 230, size=(len(object_labels), 3))

    for idx, label in enumerate(object_labels):
        mask   = (markers == label).astype(np.uint8)
        colour = colours[idx].tolist()
        overlay[mask == 1] = colour

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            x, y, w, h = cv2.boundingRect(cnts[0])
            cv2.rectangle(output, (x, y), (x + w, y + h), colour, 2)
            cx, cy = x + w // 2, y + h // 2
            cv2.putText(output, str(idx + 1), (cx - 6, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)

    return cv2.addWeighted(output, 0.6, overlay, 0.4, 0)


def draw_contour_masks(image: np.ndarray,
                        contours: list) -> np.ndarray:
    """Colour-coded contour mask overlay."""
    overlay = image.copy()
    output  = image.copy()

    rng = np.random.default_rng(42)
    colours = rng.integers(60, 230, size=(max(len(contours), 1), 3))

    for idx, cnt in enumerate(contours):
        colour = colours[idx].tolist()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        overlay[mask == 255] = colour
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), colour, 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.putText(output, str(idx + 1), (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)

    return cv2.addWeighted(output, 0.6, overlay, 0.4, 0)


def stamp_count(image: np.ndarray, count: int, method: str = "") -> np.ndarray:
    """Stamp count badge on top-left corner."""
    label = f"Count: {count}"
    if method:
        label += f"  [{method}]"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thickness = 1.0, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(image, (8, 8), (18 + tw, 22 + th), (0, 0, 0), -1)
    cv2.putText(image, label, (13, 16 + th), font, scale, (255, 255, 255), thickness)
    return image


# ──────────────────────────────────────────────
# Auto-strategy selector
# ──────────────────────────────────────────────

def best_strategy(image: np.ndarray,
                  min_area: int = 200) -> tuple:
    """
    Try all preprocessing + segmentation combos and pick the one
    that gives the most stable (median) count with most clean regions.
    Returns (annotated_image, count, strategy_name).
    """
    h, w = image.shape[:2]
    strategies = []

    for prep_name, prep_fn in [
        ("otsu",     preprocess_otsu),
        ("adaptive", preprocess_adaptive),
        ("canny",    preprocess_canny_filled),
    ]:
        binary = prep_fn(image)
        cleaned = clean_binary(binary)

        # Watershed
        try:
            markers, wcount = watershed_segment(image.copy(), cleaned)
            strategies.append((wcount, f"{prep_name}+watershed",
                                "watershed", markers, None, cleaned))
        except Exception:
            pass

        # Contour
        try:
            cnts, ccount = contour_count(cleaned, min_area)
            strategies.append((ccount, f"{prep_name}+contour",
                                "contour", None, cnts, cleaned))
        except Exception:
            pass

    if not strategies:
        return image, 0, "none"

    # Heuristic: prefer count that appears most often (mode),
    # break ties by preferring watershed
    counts = [s[0] for s in strategies]
    from collections import Counter
    mode_count = Counter(counts).most_common(1)[0][0]

    # Among strategies with mode count, prefer watershed
    candidates = [s for s in strategies if s[0] == mode_count]
    preferred = next((s for s in candidates if s[2] == "watershed"), candidates[0])

    count, name, stype, markers, cnts, _ = preferred

    if stype == "watershed":
        annotated = draw_watershed_masks(image, markers)
    else:
        annotated = draw_contour_masks(image, cnts)

    stamp_count(annotated, count, name)
    return annotated, count, name


# ──────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────

def process_image(image_path: str,
                  output_dir: str,
                  min_area: int = 200,
                  force_watershed: bool = False) -> dict:
    """Process a single image and save annotated result."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    if force_watershed:
        binary  = preprocess_otsu(image)
        cleaned = clean_binary(binary)
        markers, count = watershed_segment(image.copy(), cleaned)
        annotated = draw_watershed_masks(image, markers)
        stamp_count(annotated, count, "watershed")
        strategy = "otsu+watershed"
    else:
        annotated, count, strategy = best_strategy(image, min_area)

    os.makedirs(output_dir, exist_ok=True)
    stem     = Path(image_path).stem
    out_path = str(Path(output_dir) / f"{stem}_annotated.jpg")
    cv2.imwrite(out_path, annotated)

    print(f"[{stem}]  count={count}  strategy={strategy}  →  {out_path}")
    return {"filename": Path(image_path).name,
            "count":    count,
            "strategy": strategy,
            "output_path": out_path}


def process_directory(input_dir: str,
                       output_dir: str,
                       min_area: int = 200,
                       force_watershed: bool = False) -> list:
    exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    paths = [p for p in Path(input_dir).iterdir()
             if p.suffix.lower() in exts]
    if not paths:
        print(f"No images found in {input_dir}")
        return []

    results = []
    for p in sorted(paths):
        try:
            r = process_image(str(p), output_dir, min_area, force_watershed)
            results.append(r)
        except Exception as e:
            print(f"  ERROR {p.name}: {e}")
    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Non-AI screw/bolt counter — OpenCV")
    ap.add_argument("input",  help="Image file or directory")
    ap.add_argument("--output", default="Non_AI_output",
                    help="Output directory (default: Non_AI_output)")
    ap.add_argument("--min-area", type=int, default=200,
                    help="Min contour area to count (default: 200)")
    ap.add_argument("--force-watershed", action="store_true",
                    help="Skip auto-strategy and always use Otsu+Watershed")
    ap.add_argument("--results-json", default="",
                    help="Optional path to save JSON summary")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if os.path.isdir(args.input):
        results = process_directory(args.input, args.output,
                                    args.min_area, args.force_watershed)
    else:
        results = [process_image(args.input, args.output,
                                  args.min_area, args.force_watershed)]

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {args.results_json}")

    total = sum(r["count"] for r in results)
    print(f"\n{'─'*45}")
    print(f"Images processed : {len(results)}")
    print(f"Total objects    : {total}")
    print(f"{'─'*45}")