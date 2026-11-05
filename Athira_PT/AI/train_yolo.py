"""
Training pipeline for the AI counter (YOLOv8-seg, single class "screw").

No manual annotation exists, so training uses PSEUDO-LABELS produced automatically by the
classical pipeline in ../Non_AI (self-training):
  1. Run the Non-AI instance extractor on every training image (full resolution).
  2. Cut each image into 640x640 tiles (stride 480). Objects are small (80-220 px in a
     3000x4000 photo), so tiling keeps them at native resolution.
  3. Keep tiles whose labels are trustworthy: tiles overlapping a clump of >= 3 touching SCREWS are
     dropped (k-means splits of such clumps are unreliable). Nut piles are kept: their split is
     anchored on the visible hole of every nut, which is reliable. ~10% empty tiles are kept
     as negatives.
  4. Fine-tune the pretrained yolov8n-seg.pt (COCO) on those tiles. The network learns the object
     appearance and can then separate touching objects better than the classical splitter.

Held-out TEST images (never trained on): Screws/img5.jpg, Screws/img6.jpg.
Validation tiles come from two ScrewAndBolt images.

Usage:  python Athira_PT/AI/train_yolo.py --data data --epochs 60
"""
import os
import sys
import glob
import argparse
import random
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Non_AI"))
from count_and_mask import extract_instances  # noqa: E402

TILE, STRIDE = 640, 480
TEST_IMAGES = {"img5.jpg", "img6.jpg"}
VAL_IMAGES = {"20240713_194215.jpg", "20240713_193659.jpg"}


def tile_starts(n):
    s = list(range(0, max(1, n - TILE + 1), STRIDE))
    if s[-1] + TILE < n:
        s.append(n - TILE)
    return s


def build_dataset(data_dir, out_dir, seed=0):
    random.seed(seed)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", sp), exist_ok=True)
    counts = {"train": 0, "val": 0}
    paths = sorted(glob.glob(os.path.join(data_dir, "*", "*.jpg")))
    for p in paths:
        name = os.path.basename(p)
        if name in TEST_IMAGES:
            continue
        split = "val" if name in VAL_IMAGES else "train"
        img = cv2.imread(p)
        H, W = img.shape[:2]
        polys, bad = [], np.zeros((H, W), np.uint8)
        for part, x, y, k, reliable in extract_instances(img):
            cnts, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2) + [x, y]
            if len(c) < 3:
                continue
            polys.append(c)
            if not reliable:
                bad[y:y + part.shape[0], x:x + part.shape[1]][part > 0] = 1
        stem = os.path.splitext(name)[0]
        for ty in tile_starts(H):
            for tx in tile_starts(W):
                if bad[ty:ty + TILE, tx:tx + TILE].any():
                    continue
                lines = []
                for c in polys:
                    cx, cy = c.mean(axis=0)
                    if not (tx <= cx < tx + TILE and ty <= cy < ty + TILE):
                        continue
                    q = np.clip((c - [tx, ty]) / float(TILE), 0, 1)
                    lines.append("0 " + " ".join(f"{v:.5f}" for v in q.ravel()))
                if not lines and random.random() > 0.1:
                    continue
                tid = f"{stem}_{ty}_{tx}"
                cv2.imwrite(os.path.join(out_dir, "images", split, tid + ".jpg"),
                            img[ty:ty + TILE, tx:tx + TILE], [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(os.path.join(out_dir, "labels", split, tid + ".txt"), "w") as f:
                    f.write("\n".join(lines))
                counts[split] += 1
        print(f"{name}: {len(polys)} pseudo-labelled objects")
    print("tiles:", counts)
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir).replace(chr(92), '/')}\ntrain: images/train\nval: images/val\n"
                "names:\n  0: screw\n")
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--work", default="Athira_PT/AI/_dataset", help="generated tiles (git-ignored)")
    ap.add_argument("--base", default="yolov8n-seg.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from ultralytics import YOLO
    yaml_path = build_dataset(args.data, args.work)
    model = YOLO(args.base)
    model.train(data=yaml_path, epochs=args.epochs, imgsz=TILE, batch=args.batch, scale=0.5,
                flipud=0.5, fliplr=0.5, degrees=180, mosaic=1.0, project=os.path.join(HERE, "runs"),
                name="screw_seg", exist_ok=True, workers=0, plots=False)
    best = os.path.join(HERE, "runs", "screw_seg", "weights", "best.pt")
    print("Best weights:", best, "-> copy to Athira_PT/AI/screw_seg.pt")


if __name__ == "__main__":
    main()
