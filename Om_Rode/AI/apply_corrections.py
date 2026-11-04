"""Apply a handful of point-click corrections to the SAM auto-labels.

corrections.json (coordinates are pixels of labels_sam/images/<name>.jpg):
{
  "<image name>": {
    "add":    [[x, y], ...],                 # missed object: SAM point prompt at (x, y)
    "split":  [[x, y], ...],                 # merged mask containing (x, y): replaced by one
                                             # SAM mask per distance-transform peak (one peak
                                             # per object, other peaks as negative points)
    "delete": [[x, y], ...]                  # remove the mask containing (x, y)
  }
}
A correction is one click per object, so the human effort stays small. Rewrites the
image's YOLO label, mask file, overlay and the count in counts.csv.
"""
import csv
import json
import os
import shutil
import sys

import cv2
import numpy as np
from ultralytics import SAM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from autolabel_sam import TILE, crop, distance_peaks, draw, load_masks, mask_to_polygon, save_masks  # noqa: E402

LAB = os.path.join(HERE, "labels_sam")


def containing(masks, x, y, tol=25):
    """Mask containing (x, y); else the mask with a pixel nearest to it (within tol px).
    Centroids of V-shaped merged masks often fall outside the mask itself."""
    best, bd = None, tol ** 2 + 1
    for k, (y0, x0, m) in enumerate(masks):
        if not (y0 - tol <= y < y0 + m.shape[0] + tol and x0 - tol <= x < x0 + m.shape[1] + tol):
            continue
        ys, xs = np.nonzero(m)
        d = ((xs + x0 - x) ** 2 + (ys + y0 - y) ** 2).min()
        if d < bd:
            best, bd = k, d
    return best


def split_points(m, n):
    """n seed points inside mask m: the n most mutually distant distance-transform peaks."""
    peaks = distance_peaks(m.astype(np.uint8), 5)
    if len(peaks) < n:
        return [(py, px) for py, px in peaks]
    chosen = [peaks[0]]
    while len(chosen) < n:
        chosen.append(max(peaks, key=lambda p: min((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in chosen)))
    return chosen


def prompt(model, img, pts, labels):
    H, W = img.shape[:2]
    px, py = pts[0]
    cy, cx = min(max(py - TILE // 2, 0), max(H - TILE, 0)), min(max(px - TILE // 2, 0), max(W - TILE, 0))
    t = img[cy:cy + TILE, cx:cx + TILE]
    p = [[int(x - cx), int(y - cy)] for x, y in pts]
    r = model(t, points=[p], labels=[labels], verbose=False)[0]
    if r.masks is None or len(r.masks.data) == 0:
        return None
    m = r.masks.data[0].cpu().numpy()[:t.shape[0], :t.shape[1]] > 0.5
    if not m.any():
        return None
    y0, x0, mc = crop(m)
    return (y0 + cy, x0 + cx, mc)


def main():
    corr = json.load(open(os.path.join(HERE, "corrections.json")))
    model = SAM(os.path.join(HERE, "..", "..", "weights", "sam2.1_b.pt"))
    counts_p = os.path.join(LAB, "counts.csv")
    rows = {r["image"]: r for r in csv.DictReader(open(counts_p))}
    for name, c in corr.items():
        img = cv2.imread(os.path.join(LAB, "images", name + ".jpg"))
        H, W = img.shape[:2]
        orig = os.path.join(LAB, "masks_auto", name + ".npz")
        if not os.path.exists(orig):  # keep the untouched auto-labels; corrections are re-applied from them
            os.makedirs(os.path.dirname(orig), exist_ok=True)
            shutil.copy(os.path.join(LAB, "masks", name + ".npz"), orig)
        masks, _ = load_masks(orig)
        before = len(masks)
        for x, y in c.get("delete", []):
            k = containing(masks, x, y)
            if k is not None:
                masks.pop(k)
        for item in c.get("split", []):
            x, y, n = (item + [2])[:3]  # [x, y] or [x, y, n_objects]
            k = containing(masks, x, y)
            if k is None:
                print(f"  split point {x},{y}: no mask there")
                continue
            y0, x0, mm = masks.pop(k)
            pts = [[int(px + x0), int(py + y0)] for py, px in split_points(mm, n)]
            for i, p in enumerate(pts):
                others = [q for j, q in enumerate(pts) if j != i]
                m = prompt(model, img, [p] + others, [1] + [0] * len(others))
                if m is not None:
                    masks.append(m)
        for x, y in c.get("add", []):
            m = prompt(model, img, [[x, y]], [1])
            if m is not None:
                masks.append(m)
        save_masks(os.path.join(LAB, "masks", name + ".npz"), masks, (H, W))
        with open(os.path.join(LAB, "labels", name + ".txt"), "w") as f:
            for m in masks:
                poly = mask_to_polygon(m) / np.array([W, H])
                f.write("0 " + " ".join(f"{v:.5f}" for v in poly.ravel()) + "\n")
        cv2.imwrite(os.path.join(LAB, "overlays", name + ".jpg"), draw(img, masks))
        rows.setdefault(name, {"image": name, "recovered_by_prompt": 0})["sam_count"] = len(masks)
        print(f"{name}: {before} -> {len(masks)}")
    with open(counts_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "sam_count", "recovered_by_prompt"])
        w.writeheader()
        w.writerows(rows.values())


if __name__ == "__main__":
    main()
