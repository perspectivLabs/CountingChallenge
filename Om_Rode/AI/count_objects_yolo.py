"""AI object counting: YOLOv8s-seg instance segmentation on the whole image.

The image is resized the same way as during labelling (long side <= 2400 px) and the
model is run once at that resolution (it was trained on 1024 px tiles at the same scale,
so object sizes match). Count = number of instances; masks are overlaid.
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

MAX_SIDE = 2400


def run(model, path, conf):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    imgsz = int(np.ceil(max(img.shape[:2]) / 32) * 32)
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.5, max_det=1000, retina_masks=True, verbose=False)[0]
    masks = r.masks.data.cpu().numpy() > 0.5 if r.masks is not None else np.zeros((0, *img.shape[:2]), bool)
    ov = img.copy()
    rng = np.random.default_rng(0)
    for m in masks:
        ov[m] = (0.45 * ov[m] + 0.55 * rng.integers(40, 255, 3)).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cs, -1, (30, 30, 30), 1)
    cv2.rectangle(ov, (0, 0), (420, 60), (255, 255, 255), -1)
    cv2.putText(ov, f"Count: {len(masks)}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return ov, len(masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--weights", default="Om_Rode/AI/runs/yolov8s_seg/weights/best.pt")
    ap.add_argument("--out", default="Om_Rode/AI/outputs")
    ap.add_argument("--conf", type=float, default=0.4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)
    rows = []
    for p in sorted(glob.glob(os.path.join(args.data, "**", "*.jpg"), recursive=True)):
        ov, n = run(model, p, args.conf)
        name = os.path.basename(p)
        cv2.imwrite(os.path.join(args.out, name), ov)
        rows.append((os.path.splitext(name)[0], n))
        print(f"{name}: {n}", flush=True)
    with open(os.path.join(args.out, "counts.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("image", "count"), *rows])


if __name__ == "__main__":
    main()
