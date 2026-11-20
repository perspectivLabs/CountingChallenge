"""Build a YOLO-seg training set from the Non_AI masks (pseudo-labels).

Each image is resized to the same working size as the Non_AI pipeline
(long side 2000 px) and cut into overlapping 640 x 640 tiles. Every object
mask fully inside a tile becomes one polygon label of class 0 ("object").
Objects cut by a tile edge are kept only if most of them is inside the tile.

Held-out images (never used for training) are listed in TEST_IMAGES.

Usage:
  python make_dataset.py --data ../../data --regions ../Non_AI/output --out dataset
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np

WORK_SIDE = 2000
TILE, STRIDE = 640, 480
TEST_IMAGES = {"img5", "20240713_193659", "20240713_194206", "20240713_194316", "20240713_194621"}
VAL_IMAGES = {"img3", "20240713_193907", "20240713_194551"}


def tiles(h, w):
    ys = list(range(0, max(h - TILE, 0) + 1, STRIDE))
    xs = list(range(0, max(w - TILE, 0) + 1, STRIDE))
    if ys[-1] + TILE < h:
        ys.append(h - TILE)
    if xs[-1] + TILE < w:
        xs.append(w - TILE)
    return [(y, x) for y in ys for x in xs]


def polygons(regions, y, x):
    crop = regions[y:y + TILE, x:x + TILE]
    lines = []
    for r in np.unique(crop):
        if r == 0:
            continue
        inside = int((crop == r).sum())
        total = int((regions == r).sum())
        if inside < 0.6 * total or inside < 30:
            continue
        cnts, _ = cv2.findContours((crop == r).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        pts = " ".join(f"{px / TILE:.5f} {py / TILE:.5f}" for px, py in c)
        lines.append("0 " + pts)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data")
    ap.add_argument("--regions", default="../Non_AI/output")
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()
    shutil.rmtree(args.out, ignore_errors=True)
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split))
        os.makedirs(os.path.join(args.out, "labels", split))

    for path in sorted(glob.glob(os.path.join(args.data, "*", "*.jpg"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in TEST_IMAGES:
            continue
        split = "val" if name in VAL_IMAGES else "train"
        regions = np.load(os.path.join(args.regions, name + "_regions.npy")).astype(np.int32)
        img = cv2.imread(path)
        img = cv2.resize(img, (regions.shape[1], regions.shape[0]), interpolation=cv2.INTER_AREA)
        n = 0
        for (y, x) in tiles(*regions.shape):
            lines = polygons(regions, y, x)
            if not lines:
                continue
            stem = f"{name}_{y}_{x}"
            cv2.imwrite(os.path.join(args.out, "images", split, stem + ".jpg"),
                        img[y:y + TILE, x:x + TILE], [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(args.out, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            n += 1
        print(f"{name:22s} {split:5s} {n} tiles")

    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: object\n")


if __name__ == "__main__":
    main()
