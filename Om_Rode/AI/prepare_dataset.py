"""Build a tiled YOLOv8-seg dataset from the SAM auto-labels.

Split is by *image* (never by tile) so test images are completely unseen:
  test : one image from every scene type (held out, used only for the final accuracy table)
  val  : two images (model selection / early stopping)
  train: the rest

Training images are cut into 1024 px tiles (stride 512). An object is labelled in a tile
if >= 40 % of its mask is visible; the visible part is the polygon.
Val/test images are kept whole: the model is run on the full image at inference time.
"""
import argparse
import os
import shutil

import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autolabel_sam import load_masks, window  # noqa: E402

TEST = ["img4", "20240713_193659", "20240713_193907", "20240713_194256", "20240713_194621"]
VAL = ["img6", "20240713_194206"]
TILE, STRIDE = 1024, 512


def polys(c, oy=0, ox=0):
    """Polygon of a crop (y0, x0, m) in the coordinates of a window starting at (oy, ox)."""
    y0, x0, m = c
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x0 - ox, y0 - oy))
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 20:
        return None
    return cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)


def write(img, masks, out_img, out_lbl, oy=0, ox=0):
    H, W = img.shape[:2]
    cv2.imwrite(out_img, img)
    with open(out_lbl, "w") as f:
        for c in masks:
            p = polys(c, oy, ox)
            if p is None or len(p) < 3:
                continue
            p = p / np.array([W, H])
            f.write("0 " + " ".join(f"{v:.5f}" for v in p.ravel()) + "\n")


def tile_starts(n):
    s = list(range(0, max(n - TILE, 0) + 1, STRIDE)) or [0]
    if s[-1] + TILE < n:
        s.append(n - TILE)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="Om_Rode/AI/labels_sam")
    ap.add_argument("--out", default="Om_Rode/AI/dataset")
    args = ap.parse_args()
    shutil.rmtree(args.out, ignore_errors=True)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(args.out, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, sp, "labels"), exist_ok=True)
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(args.labels, "images")))
    n_tiles = 0
    for name in names:
        img = cv2.imread(os.path.join(args.labels, "images", name + ".jpg"))
        masks, _ = load_masks(os.path.join(args.labels, "masks", name + ".npz"))
        sp = "test" if name in TEST else "val" if name in VAL else "train"
        if sp != "train":
            write(img, masks, *(os.path.join(args.out, sp, d, name + e) for d, e in (("images", ".jpg"), ("labels", ".txt"))))
            continue
        H, W = img.shape[:2]
        tw, th = min(TILE, W), min(TILE, H)
        for y in tile_starts(H):
            for x in tile_starts(W):
                sel = []
                for c in masks:
                    v = window(c, y, x, y + th, x + tw)
                    a = c[2].sum()
                    if v.sum() >= 0.4 * a:
                        sel.append((y, x, v))  # visible part, in window coordinates
                write(img[y:y + th, x:x + tw], sel,
                      os.path.join(args.out, sp, "images", f"{name}_{y}_{x}.jpg"),
                      os.path.join(args.out, sp, "labels", f"{name}_{y}_{x}.txt"), y, x)
                n_tiles += 1
    root = os.path.abspath(args.out)
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"path: {root}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: object\n")
    print(f"train tiles: {n_tiles}; val: {len(VAL)} images; test: {len(TEST)} images")


if __name__ == "__main__":
    main()
