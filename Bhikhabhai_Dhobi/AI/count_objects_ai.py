"""AI object counter: YOLO11-seg trained in Colab on Non_AI pseudo-labels.

The model was trained on 640 x 640 tiles cut from images resized to a long
side of 2000 px. Every photo is resized to that same working size and the
network runs once on the whole image (imgsz=2016), so objects appear at
exactly the scale seen in training and no tile seams cut masks in half.

(SAHI sliced inference was tried first; it cut masks at slice borders and
found fewer objects in dense clusters, so it was dropped - see README.)

Count = number of predicted instances.

Usage:
  python count_objects_ai.py --weights weights/best.pt --data ../../data --out output
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

WORK_SIDE = 2000   # same working size as Non_AI and as the training tiles
IMGSZ = 2016       # multiple of 32 that holds the whole working image


def load(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = WORK_SIDE / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def drop_fragments(masks, max_covered=0.5):
    """Mask-level duplicate removal. Box NMS misses partial duplicates of long
    diagonal screws (a fragment's box overlaps the full screw's box only a
    little), so a mask is dropped when most of it lies inside larger masks
    that were already kept."""
    order = np.argsort([-m.sum() for m in masks])
    kept, union = [], np.zeros_like(masks[0]) if masks else None
    for i in order:
        m = masks[i]
        if (m & union).sum() / max(m.sum(), 1) >= max_covered:
            continue
        kept.append(m)
        union |= m
    return kept


def overlay(img, masks, title):
    rng = np.random.default_rng(11)
    out = img.copy()
    for m in masks:
        color = rng.integers(40, 255, 3)
        out[m] = (0.45 * out[m] + 0.55 * color).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
    cv2.rectangle(out, (0, 0), (900, 70), (0, 0, 0), -1)
    cv2.putText(out, title, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--data", default="../../data")
    ap.add_argument("--out", default="output")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)
    rows = []
    for path in sorted(glob.glob(os.path.join(args.data, "*", "*.jpg"))):
        name = os.path.splitext(os.path.basename(path))[0]
        img = load(path)
        r = model.predict(img, imgsz=IMGSZ, conf=args.conf, max_det=1000,
                          device=args.device, verbose=False)[0]
        masks = []
        if len(r):
            # masks come at network resolution: resize each to the image
            m = r.masks.data.cpu().numpy()
            masks = drop_fragments([cv2.resize(x, (img.shape[1], img.shape[0]),
                                               interpolation=cv2.INTER_NEAREST) > 0.5 for x in m])
        n = len(masks)
        cv2.imwrite(os.path.join(args.out, name + "_overlay.jpg"),
                    overlay(img, masks, f"{name}: {n} objects"), [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append((name, n))
        print(f"{name:22s} {n:5d}")
    with open(os.path.join(args.out, "counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "count"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
