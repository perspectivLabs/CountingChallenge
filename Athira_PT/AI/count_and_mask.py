"""
AI Object Counting and Mask Overlay Pipeline
Author: Athira PT
Task: Count objects (screws, bolts, fasteners) and overlay precise instance segmentation masks
      with > 95% accuracy using a YOLOv8-seg model fine-tuned on pseudo-labels (see train_yolo.py).
"""

import os
import sys
import csv
import glob
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO


TILE, STRIDE = 640, 320   # overlap (320) > largest object (~250 px) so every object is whole in some tile


def get_ai_model(weights="screw_seg.pt"):
    """Loads the fine-tuned YOLOv8-seg weights (produced by train_yolo.py)."""
    device = 0 if torch.cuda.is_available() else "cpu"
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (weights, os.path.join(here, weights), os.path.join(here, "runs", "screw_seg", "weights", "best.pt")):
        if os.path.exists(c):
            return YOLO(c), device
    raise FileNotFoundError("Trained weights not found. Run train_yolo.py first (see README).")


def _starts(n):
    s = list(range(0, max(1, n - TILE + 1), STRIDE))
    if s[-1] + TILE < n:
        s.append(n - TILE)
    return s


def tiled_predict(model, img, device, conf=0.25, iou=0.6):
    """Runs the model on overlapping native-resolution tiles. Returns [(score, contour_xy)]."""
    H, W = img.shape[:2]
    dets = []
    for ty in _starts(H):
        for tx in _starts(W):
            tile = img[ty:ty + TILE, tx:tx + TILE]
            r = model(tile, device=device, imgsz=TILE, conf=conf, iou=iou, retina_masks=True, verbose=False)[0]
            if r.masks is None:
                continue
            for poly, sc in zip(r.masks.xy, r.boxes.conf.cpu().numpy()):
                if len(poly) < 3:
                    continue
                th, tw = tile.shape[:2]
                # drop objects cut by an INNER tile edge: the neighbouring tile sees them whole
                x0, y0 = poly.min(axis=0)
                x1, y1 = poly.max(axis=0)
                cut = ((x0 < 2 and tx > 0) or (y0 < 2 and ty > 0) or
                       (x1 > tw - 3 and tx + tw < W) or (y1 > th - 3 and ty + th < H))
                if cut:
                    continue
                dets.append((float(sc), (poly + [tx, ty]).astype(np.int32)))
    return dets


def merge_duplicates(dets, shape, iomin=0.9):
    """Greedy cross-tile NMS on masks: keep highest score, drop a mask only if > iomin of it is already covered
    (high value: touching or overlapping screws must survive, only near-identical duplicates from tile overlaps go)."""
    H, W = shape
    dets = sorted(dets, key=lambda d: -d[0])
    taken = np.zeros((H, W), np.uint8)
    kept = []
    for sc, poly in dets:
        x, y, w, h = cv2.boundingRect(poly)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [poly - [x, y]], 1)
        area = int(m.sum())
        if area < 60:
            continue
        inter = int((taken[y:y + h, x:x + w] & m).sum())
        if inter / area > iomin:
            continue
        taken[y:y + h, x:x + w] |= m
        kept.append((sc, poly))
    return kept


def detect_items_ai(image_input, model=None, conf=0.25, overlay_alpha=0.45, device=None):
    """
    Detects, segments and counts objects with the fine-tuned YOLOv8-seg model (tiled inference).

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

    if model is None:
        model, device = get_ai_model()
    elif device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    kept = merge_duplicates(tiled_predict(model, img, device, conf=conf), (orig_h, orig_w))

    composite_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    mask_layer = np.zeros_like(img)
    overlay = img.copy()
    colors = [(0, 255, 0), (0, 215, 255), (255, 128, 0), (0, 165, 255), (255, 0, 128), (50, 205, 50)]
    scale_factor = max(orig_h, orig_w) / 1080.0
    items = []

    for idx, (sc, poly) in enumerate(kept, start=1):
        color = colors[(idx - 1) % len(colors)]
        cv2.fillPoly(composite_mask, [poly], 255)
        cv2.fillPoly(mask_layer, [poly], color)
        cv2.polylines(overlay, [poly], True, color, max(2, int(round(2 * scale_factor))))
        x, y, w, h = cv2.boundingRect(poly)
        M = cv2.moments(poly)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] else y + h // 2
        cv2.circle(overlay, (cx, cy), max(3, int(round(3.5 * scale_factor))), (0, 0, 255), -1)
        cv2.putText(overlay, f"#{idx}", (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.35, 0.45 * scale_factor), color, max(1, int(round(1.5 * scale_factor))), cv2.LINE_AA)
        items.append({"id": idx, "bbox": [x, y, w, h], "centroid": (cx, cy),
                      "area": float(cv2.contourArea(poly)), "score": sc})

    cv2.addWeighted(mask_layer, overlay_alpha, overlay, 1.0 - overlay_alpha, 0, overlay)
    total = len(items)
    banner_height = max(55, int(round(65 * scale_factor)))
    banner = np.full((banner_height, orig_w, 3), 25, dtype=np.uint8)
    cv2.putText(banner, f"AI Object Count: {total}  |  Model: fine-tuned YOLOv8-seg (tiled)",
                (20, int(banner_height * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, max(0.55, 0.75 * scale_factor),
                (0, 255, 255), max(1, int(round(2 * scale_factor))), cv2.LINE_AA)

    return {"count": total, "items": items, "annotated_image": np.vstack([banner, overlay]),
            "overlay_image": overlay, "binary_mask": composite_mask, "original_image": img}


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


def evaluate_dataset_ai(dataset_dir, output_dir=None, conf=0.25):
    """Evaluates the AI model across a directory of images against the hand-counted ground truth."""
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
    print(f"AI EVALUATION: {os.path.basename(dataset_dir)} ({len(image_paths)} images)")
    print(f"{'='*78}")
    print(f"{'Filename':<30} | {'Detected':<9} | {'Ground Truth':<12} | {'Accuracy':<9}")
    print(f"{'-'*78}")
    model, device = get_ai_model()

    results = []
    for img_p in image_paths:
        fname = os.path.basename(img_p)
        stem = os.path.splitext(fname)[0]
        res = detect_items_ai(img_p, model=model, conf=conf, device=device)
        gt = gt_all.get(stem)
        acc = count_accuracy(res["count"], gt) if gt else None
        results.append({"filename": fname, "count": res["count"], "gt": gt, "acc": acc})
        acc_txt = f"{acc:.1f}%" if acc is not None else "-"
        print(f"{fname:<30} | {res['count']:<9} | {str(gt) if gt else '-':<12} | {acc_txt}")

        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"ai_detected_{stem}.jpg"), res["annotated_image"])
            cv2.imwrite(os.path.join(output_dir, f"ai_mask_{stem}.png"), res["binary_mask"])

    print(f"{'-'*78}")
    accs = [r["acc"] for r in results if r["acc"] is not None]
    if accs:
        ok = sum(a > 95.0 for a in accs)
        print(f"Summary: Images={len(results)} | Mean Accuracy={np.mean(accs):.2f}% | "
              f"Images > 95%: {ok}/{len(accs)} | Worst={min(accs):.1f}%")
    print(f"{'='*78}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="AI Object Counting and Mask Overlay using fine-tuned YOLOv8-seg")
    parser.add_argument("--input", type=str, default="data/Screws", help="Path to image file or directory")
    parser.add_argument("--output_dir", type=str, default="Athira_PT/AI/output", help="Directory to save output visualizations")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        evaluate_dataset_ai(args.input, output_dir=args.output_dir, conf=args.conf)
    elif os.path.isfile(args.input):
        res = detect_items_ai(args.input, conf=args.conf)
        print(f"\nProcessed {args.input}: Detected Objects={res['count']}")
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            fname = os.path.basename(args.input)
            out_img = os.path.join(args.output_dir, f"ai_detected_{fname}")
            out_mask = os.path.join(args.output_dir, f"ai_mask_{os.path.splitext(fname)[0]}.png")
            cv2.imwrite(out_img, res["annotated_image"])
            cv2.imwrite(out_mask, res["binary_mask"])
            print(f"Saved annotated overlay to: {out_img}")
            print(f"Saved binary mask to: {out_mask}")
    else:
        print(f"Error: Input path '{args.input}' does not exist.")


if __name__ == "__main__":
    main()
