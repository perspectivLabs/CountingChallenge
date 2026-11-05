"""
Non-AI Object Counting and Mask Overlay Pipeline
Author: Athira PT
Task: Count objects (screws, bolts, fasteners) and overlay segmentation masks
      with > 95% accuracy using classical Computer Vision (OpenCV).

Pipeline (works for both the `Screws` and `ScrewAndBolt` datasets):
  1. Background-subtracted silhouette segmentation at FULL resolution.
  2. Connected components -> every blob is either one object or a touching group.
  3. Estimate the area of ONE object from the smallest (isolated) blobs of the image.
  4. Blob area / single-object area -> number of objects in the blob.
  5. Groups are split into that many parts with k-means on the pixel coordinates.
"""

import os
import csv
import glob
import argparse
import cv2
import numpy as np

MIN_ABS_AREA = 400          # px, absolute noise floor (full resolution)
MIN_REL_AREA = 0.45         # blobs smaller than 45% of a single object = noise
BORDER_REL_AREA = 0.60      # blobs touching the image border must be >= 60% of an object
                            # (rejects dark vignette corners / partially visible debris)
SINGLES_FRACTION = 0.65     # smallest 65% of blobs -> median = single-object area (tuned on ground truth)
MIN_HOLE_AREA = 60          # px, smallest hole considered a nut hole
OVERLAP_GAIN = 1.00         # object count of a blob = round(area / single-object area)


def segment_foreground(img):
    """Binary silhouette of objects: local background model + Otsu-scaled threshold."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    h, w = gray.shape
    # Background = smooth surface (degree-6 polynomial in x,y) fitted robustly to the image:
    # iteratively refit using only pixels that are NOT darker than the current fit, so objects
    # (even huge piles) never pull the surface down, while vignetting / lighting gradients are followed.
    f = 0.05
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA).astype(np.float64)
    sh, sw = small.shape
    yy, xx = np.mgrid[0:sh, 0:sw]
    xn, yn = xx / max(sw - 1, 1) * 2 - 1, yy / max(sh - 1, 1) * 2 - 1
    terms = [(xn ** i) * (yn ** j) for i in range(7) for j in range(7 - i)]
    A = np.stack([t.ravel() for t in terms], axis=1)
    z = small.ravel()
    use = np.ones_like(z, dtype=bool)
    for _ in range(8):
        coef, *_ = np.linalg.lstsq(A[use], z[use], rcond=None)
        fit = A @ coef
        use = (z - fit) > -15.0          # keep background-like pixels (not clearly darker than the fit)
        if use.sum() < 0.2 * z.size:
            break
    bg_small = np.clip(fit.reshape(sh, sw), 0, 255).astype(np.float32)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    gray = gray.astype(np.float32)
    # Objects are darker than the background (backlit / light table).
    diff = np.clip(bg - gray, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (diff > max(25, 0.6 * otsu)).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def estimate_single_area(areas):
    """Area of one object = median of the smallest blobs (those are isolated objects)."""
    areas = np.sort(np.asarray(areas, dtype=np.float64))
    k = max(3, int(len(areas) * SINGLES_FRACTION))
    return float(np.median(areas[:k]))


def split_blob(blob_mask, k):
    """Split a binary blob into k parts using k-means on pixel coordinates."""
    ys, xs = np.nonzero(blob_mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, _ = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    parts = []
    for i in range(k):
        part = np.zeros_like(blob_mask)
        part[ys[labels == i], xs[labels == i]] = 255
        parts.append(part)
    return parts


def find_holes(mask, labels):
    """
    Enclosed background holes of the silhouette (the hole of every nut, even when nuts touch).
    Returns {blob_label: [(cx, cy), ...]} keeping only round holes of a consistent size.
    """
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return {}
    cand = []
    for i, c in enumerate(cnts):
        parent = hier[0][i][3]
        if parent == -1:
            continue
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        if area < MIN_HOLE_AREA or per == 0:
            continue
        circ = 4 * np.pi * area / (per * per)
        M = cv2.moments(c)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        px, py = cnts[parent][0][0]
        cand.append((area, circ, cx, cy, int(labels[py, px])))
    round_areas = [a for a, circ, *_ in cand if circ > 0.7]
    if len(round_areas) < 3:
        return {}
    ref = float(np.median(round_areas))
    holes = {}
    for a, circ, cx, cy, lab in cand:
        if circ > 0.7 and 0.5 * ref <= a <= 2.0 * ref:
            holes.setdefault(lab, []).append((cx, cy))
    return holes


def split_by_seeds(blob_mask, seeds):
    """Assign every blob pixel to its nearest seed (Voronoi split) -> one mask per seed."""
    ys, xs = np.nonzero(blob_mask)
    d = (xs[:, None] - seeds[None, :, 0]) ** 2 + (ys[:, None] - seeds[None, :, 1]) ** 2
    nearest = d.argmin(axis=1)
    parts = []
    for i in range(len(seeds)):
        part = np.zeros_like(blob_mask)
        part[ys[nearest == i], xs[nearest == i]] = 255
        if part.any():
            parts.append(part)
    return parts


def extract_instances(img):
    """Returns list of (mask_crop, x, y, k, reliable): one entry per object, k = objects in its source blob,
    reliable = True when the split is anchored on nut holes or the blob is a single object."""
    orig_h, orig_w = img.shape[:2]
    # Steps 1-2: silhouette + connected components
    mask = segment_foreground(img)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blob_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_ABS_AREA]

    # Step 3: area of a single object
    single_area = estimate_single_area([stats[i, cv2.CC_STAT_AREA] for i in blob_ids]) if blob_ids else 1.0

    # Nut holes: robust per-object evidence inside piles of touching nuts
    holes = find_holes(mask, labels)

    # Steps 4-5: count per blob, split groups
    instances = []
    for i in blob_ids:
        x, y, w, h, area = stats[i]
        if area < MIN_REL_AREA * single_area:
            continue
        touches_border = x == 0 or y == 0 or x + w >= orig_w or y + h >= orig_h
        if touches_border and area < BORDER_REL_AREA * single_area:
            continue
        blob = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        hs = holes.get(i, [])
        if len(hs) >= 2:
            # >= 2 nut holes: one nut per hole, split pixels to the nearest hole
            seeds = np.array([(hx - x, hy - y) for hx, hy in hs], dtype=np.float64)
            parts = split_by_seeds(blob, seeds)
            k = len(parts)
            reliable = True
        else:
            k = max(1, int(round(OVERLAP_GAIN * area / single_area)))
            parts = split_blob(blob, k) if k > 1 else [blob]
            reliable = k <= 2
        for p in parts:
            instances.append((p, x, y, k, reliable))

    return instances


def detect_items_non_ai(image_input, overlay_alpha=0.45):
    """
    Detects, segments and counts objects with classical Computer Vision.

    Returns dict: count, items, annotated_image, overlay_image, binary_mask, original_image
    """
    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image not found at: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not read image file: {image_input}")
    else:
        img = image_input.copy()

    orig_h, orig_w = img.shape[:2]

    instances = [(p, x, y) for p, x, y, *_ in extract_instances(img)]

    # Step 6: visualisation
    binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    overlay = img.copy()
    mask_layer = np.zeros_like(img)
    colors = [(0, 255, 0), (0, 215, 255), (255, 128, 0), (0, 165, 255), (255, 0, 128), (50, 205, 50)]
    scale_factor = max(orig_h, orig_w) / 1080.0
    items = []

    for idx, (p, x0, y0) in enumerate(instances, start=1):
        color = colors[(idx - 1) % len(colors)]
        cnts, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea) + [x0, y0]
        full = [c + [x0, y0] for c in cnts]
        cv2.drawContours(binary_mask, full, -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask_layer, full, -1, color, thickness=cv2.FILLED)
        cv2.drawContours(overlay, full, -1, color, thickness=max(2, int(round(2 * scale_factor))))

        bx, by, bw, bh = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else bx + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] else by + bh // 2
        cv2.circle(overlay, (cx, cy), max(3, int(round(3.5 * scale_factor))), (0, 0, 255), -1)
        cv2.putText(overlay, f"#{idx}", (bx, max(18, by - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.35, 0.45 * scale_factor), color, max(1, int(round(1.5 * scale_factor))), cv2.LINE_AA)
        items.append({"id": idx, "bbox": [int(bx), int(by), int(bw), int(bh)],
                      "centroid": (cx, cy), "area": float(cv2.countNonZero(p))})

    cv2.addWeighted(mask_layer, overlay_alpha, overlay, 1.0 - overlay_alpha, 0, overlay)

    total = len(items)
    banner_height = max(55, int(round(65 * scale_factor)))
    banner = np.full((banner_height, orig_w, 3), 25, dtype=np.uint8)
    cv2.putText(banner, f"Non-AI Object Count: {total}  |  Method: Silhouette + Area-based Group Splitting",
                (20, int(banner_height * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, max(0.55, 0.75 * scale_factor),
                (0, 255, 255), max(1, int(round(2 * scale_factor))), cv2.LINE_AA)

    return {"count": total, "items": items, "annotated_image": np.vstack([banner, overlay]),
            "overlay_image": overlay, "binary_mask": binary_mask, "original_image": img}


def load_ground_truth(path=None):
    """Hand-counted ground truth: {image_stem: count} from Athira_PT/ground_truth.csv."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ground_truth.csv")
    gt = {}
    if os.path.isfile(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                gt[row["image"]] = int(row["count"])
    return gt


def count_accuracy(pred, gt):
    """Counting accuracy in percent: 100 * (1 - |pred - gt| / gt)."""
    return max(0.0, 1.0 - abs(pred - gt) / gt) * 100.0


def evaluate_dataset(dataset_dir, output_dir=None):
    """Evaluates the Non-AI method across a directory of images against the hand-counted ground truth."""
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext.upper())))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No images found in {dataset_dir}")
        return []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    gt_all = load_ground_truth()
    print(f"\n{'='*78}")
    print(f"NON-AI EVALUATION: {os.path.basename(dataset_dir)} ({len(image_paths)} images)")
    print(f"{'='*78}")
    print(f"{'Filename':<30} | {'Detected':<9} | {'Ground Truth':<12} | {'Accuracy':<9}")
    print(f"{'-'*78}")

    results = []
    for img_p in image_paths:
        fname = os.path.basename(img_p)
        stem = os.path.splitext(fname)[0]
        res = detect_items_non_ai(img_p)
        gt = gt_all.get(stem)
        acc = count_accuracy(res["count"], gt) if gt else None
        results.append({"filename": fname, "count": res["count"], "gt": gt, "acc": acc})
        acc_txt = f"{acc:.1f}%" if acc is not None else "-"
        print(f"{fname:<30} | {res['count']:<9} | {str(gt) if gt else '-':<12} | {acc_txt}")

        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"detected_{stem}.jpg"), res["annotated_image"])
            cv2.imwrite(os.path.join(output_dir, f"mask_{stem}.png"), res["binary_mask"])

    print(f"{'-'*78}")
    accs = [r["acc"] for r in results if r["acc"] is not None]
    if accs:
        ok = sum(a > 95.0 for a in accs)
        print(f"Summary: Images={len(results)} | Mean Accuracy={np.mean(accs):.2f}% | "
              f"Images > 95%: {ok}/{len(accs)} | Worst={min(accs):.1f}%")
    print(f"{'='*78}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Non-AI Object Counting and Mask Overlay using OpenCV")
    parser.add_argument("--input", type=str, default="data/Screws", help="Path to image file or directory")
    parser.add_argument("--output_dir", type=str, default="Athira_PT/Non_AI/output", help="Directory to save output visualizations")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        evaluate_dataset(args.input, output_dir=args.output_dir)
    elif os.path.isfile(args.input):
        res = detect_items_non_ai(args.input)
        print(f"\nProcessed {args.input}: Detected Objects={res['count']}")
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            fname = os.path.basename(args.input)
            out_img = os.path.join(args.output_dir, f"detected_{fname}")
            out_mask = os.path.join(args.output_dir, f"mask_{os.path.splitext(fname)[0]}.png")
            cv2.imwrite(out_img, res["annotated_image"])
            cv2.imwrite(out_mask, res["binary_mask"])
            print(f"Saved annotated overlay to: {out_img}")
            print(f"Saved binary mask to: {out_mask}")
    else:
        print(f"Error: Input path '{args.input}' does not exist.")


if __name__ == "__main__":
    main()
