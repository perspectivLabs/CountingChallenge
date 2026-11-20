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
from ultralytics.models.yolo.segment import predict as _seg_predict
from ultralytics.utils.ops import scale_image

# Keep instance masks at the network's prototype resolution (1/4 of the input)
# instead of upsampling all of them to 2016 px at once: for a 354-nut image
# that upsampled float tensor is ~4.5 GB and crashes an 8 GB laptop. Each mask
# is upscaled individually when it is drawn. Counts come from the detections
# and are not affected.
_process_mask = _seg_predict.ops.process_mask
_seg_predict.ops.process_mask = lambda proto, m, b, shape, upsample=False: _process_mask(
    proto, m, b, shape, upsample=False)

WORK_SIDE = 2000   # same working size as Non_AI and as the training tiles
IMGSZ = 2016       # multiple of 32 that holds the whole working image


def load(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = WORK_SIDE / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def instance_masks(result, shape):
    """Yield one boolean full-size mask at a time (low-res mask -> image
    size; scale_image undoes the letterbox padding)."""
    for m in result.masks.data:
        small = (m > 0.5).cpu().numpy().astype(np.uint8)[..., None]
        big = cv2.resize(small[..., 0], (small.shape[1] * 4, small.shape[0] * 4), interpolation=cv2.INTER_LINEAR)
        yield scale_image(big[..., None], shape[:2])[..., 0] > 0


def overlay(img, masks, title):
    rng = np.random.default_rng(11)
    out = img.copy()
    for m in masks:
        out[m] = (0.45 * out[m] + 0.55 * rng.integers(40, 255, 3)).astype(np.uint8)
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
    ap.add_argument("--conf", type=float, default=0.25,
                    help="chosen on the non-test images with ground truth (see README)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)
    with open(os.path.join(args.out, "counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "count"])
        for path in sorted(glob.glob(os.path.join(args.data, "*", "*.jpg"))):
            name = os.path.splitext(os.path.basename(path))[0]
            img = load(path)
            r = model.predict(img, imgsz=IMGSZ, conf=args.conf, max_det=1000,
                              device=args.device, verbose=False)[0]
            n = len(r)
            cv2.imwrite(os.path.join(args.out, name + "_overlay.jpg"),
                        overlay(img, instance_masks(r, img.shape) if n else [], f"{name}: {n} objects"),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            w.writerow([name, n])
            f.flush()
            print(f"{name:22s} {n:5d}", flush=True)


if __name__ == "__main__":
    main()
