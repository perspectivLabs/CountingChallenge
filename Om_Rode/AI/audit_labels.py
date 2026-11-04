"""Audit SAM auto-labels with independent classical signals so a human only has to look
at flagged spots (minimal human-in-the-loop).

Flags per image:
  * miss   : foreground blob (classical threshold) not covered by any SAM mask
  * merge  : SAM mask much larger than a typical object (may contain 2+ objects)
  * holes  : nut images - number of round holes vs number of masks (should be equal)
Also reports the Non-AI count for comparison. Writes audit.csv and crops of flagged spots.
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "Non_AI"))
from autolabel_sam import load_masks  # noqa: E402
from count_objects import find_round_holes, foreground_mask  # noqa: E402

LAB = os.path.join(HERE, "labels_sam")


def main():
    out_dir = os.path.join(LAB, "flags")
    os.makedirs(out_dir, exist_ok=True)
    nonai = {}
    p = os.path.join(HERE, "..", "Non_AI", "outputs", "counts.csv")
    if os.path.exists(p):
        with open(p) as f:
            nonai = {os.path.splitext(os.path.basename(r["image"]))[0]: int(r["count"]) for r in csv.DictReader(f)}
    rows = []
    for ip in sorted(glob.glob(os.path.join(LAB, "images", "*.jpg"))):
        name = os.path.splitext(os.path.basename(ip))[0]
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        masks, _ = load_masks(os.path.join(LAB, "masks", name + ".npz"))
        s2 = 1600 / max(H, W)
        small = cv2.resize(img, (int(W * s2), int(H * s2)), interpolation=cv2.INTER_AREA) if s2 < 1 else img
        _, norm, valid = foreground_mask(small)
        # raw darkness (no hole filling), so background gaps enclosed by parts are not "objects"
        t_otsu, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fg = (cv2.resize((norm < t_otsu).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR) > 0) & \
            (cv2.resize(valid, (W, H), interpolation=cv2.INTER_NEAREST) > 0)
        covered = np.zeros((H, W), bool)
        for y0, x0, m in masks:
            covered[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] |= m
        areas = np.array([m.sum() for _, _, m in masks])
        med = float(np.median(areas)) if len(areas) else 1.0

        flags = []
        rest = cv2.morphologyEx((fg & ~covered).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(rest, 8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 0.2 * med:
                continue
            x, y, w, h = st[i, :4]
            comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
            # a real missed object is a thick blob; gaps/shadows between parts are slivers
            if cv2.distanceTransform(comp, cv2.DIST_L2, 5).max() > 0.25 * np.sqrt(med):
                flags.append(("miss", int(cen[i][0]), int(cen[i][1])))
        for (y0, x0, m), a in zip(masks, areas):
            if a > 1.6 * med:
                ys, xs = np.where(m)
                flags.append(("merge", int(xs.mean() + x0), int(ys.mean() + y0)))

        # holes (nuts) - at the 1600 px scale used by the Non-AI pipeline
        _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        holes, _ = find_round_holes(th)
        n_holes = len(holes)

        crops = []
        r = int(2.2 * np.sqrt(med))
        for k, (kind, x, y) in enumerate(flags):
            ov = cv2.imread(os.path.join(LAB, "overlays", name + ".jpg"))
            c = ov[max(y - r, 0):y + r, max(x - r, 0):x + r].copy()
            cv2.circle(c, (min(r, x), min(r, y)), int(r * 0.6), (0, 0, 255) if kind == "miss" else (255, 0, 0), 3)
            fn = f"{name}_{k}_{kind}.jpg"
            cv2.imwrite(os.path.join(out_dir, fn), c)
            crops.append(fn)
        row = dict(image=name, sam=len(masks), non_ai=nonai.get(name, ""), holes=n_holes,
                   miss=sum(f[0] == "miss" for f in flags), merge=sum(f[0] == "merge" for f in flags),
                   crops=";".join(crops), points=";".join(f"{k}:{x},{y}" for k, x, y in flags))
        rows.append(row)
        print(row["image"], "sam", row["sam"], "non_ai", row["non_ai"], "holes", n_holes, "miss", row["miss"], "merge", row["merge"], flush=True)
    with open(os.path.join(LAB, "audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
