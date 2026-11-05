"""
Build a YOLO-seg dataset (640px tiles) from images + instance label PNGs.

Labels come from model-assisted labelling: the Non-AI pipeline writes 16-bit instance
masks (`--save-masks`); these can be corrected by hand (CVAT / Roboflow) before training.
Held-out test images (common.TEST_IMAGES) are excluded from train/val.

  python make_dataset.py --images <data_dir> --masks ../Non_AI/results/masks --out dataset
"""
import argparse, glob, os, random
import cv2
import numpy as np
from common import load, tiles, TEST_IMAGES, TILE

def mask_to_polygon(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs: return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 15: return None
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    return c if len(c) >= 3 else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()
    random.seed(0)
    for s in ("train", "val"):
        os.makedirs(f"{a.out}/images/{s}", exist_ok=True); os.makedirs(f"{a.out}/labels/{s}", exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.images, "**", "*.jpg"), recursive=True))
    n_tiles = {"train": 0, "val": 0}
    for p in paths:
        name = os.path.basename(p)
        if name in TEST_IMAGES: continue
        mp = os.path.join(a.masks, os.path.splitext(name)[0] + ".png")
        if not os.path.exists(mp): print("no mask for", name); continue
        im = load(p); lab = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        assert lab.shape == im.shape[:2], (name, lab.shape, im.shape)
        H, W = lab.shape
        for (y0, x0) in tiles(H, W):
            t_im = im[y0:y0+TILE, x0:x0+TILE]; t_lab = lab[y0:y0+TILE, x0:x0+TILE]
            ids = [i for i in np.unique(t_lab) if i]
            if not ids and random.random() > 0.15: continue      # keep a few empty tiles as negatives
            th, tw = t_lab.shape
            lines = []
            for i in ids:
                m = t_lab == i
                # skip slivers of objects mostly outside this tile
                if m.sum() < 0.35 * (lab == i).sum(): continue
                poly = mask_to_polygon(m)
                if poly is None: continue
                xy = " ".join(f"{x/tw:.5f} {y/th:.5f}" for x, y in poly)
                lines.append(f"0 {xy}")
            split = "val" if random.random() < a.val_frac else "train"
            stem = f"{os.path.splitext(name)[0]}_{y0}_{x0}"
            cv2.imwrite(f"{a.out}/images/{split}/{stem}.jpg", t_im)
            with open(f"{a.out}/labels/{split}/{stem}.txt", "w") as f: f.write("\n".join(lines))
            n_tiles[split] += 1
    with open(f"{a.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: object\n")
    print("tiles:", n_tiles)

if __name__ == "__main__":
    main()
