"""AI object counting with FastSAM (zero-shot instance segmentation).

FastSAM is a YOLOv8-seg model trained on Meta's SA-1B "segment anything"
dataset (11M images, 1.1B masks). It segments *every* object in an image
without being told the class, so no training on this dataset is needed.

Pipeline (per image):
  1. Resize so the long side is WORK_SIZE px.
  2. Slice into overlapping TILE x TILE tiles (objects in the dense piles
     are small, so running on tiles keeps them large enough for the model).
     Each tile -> FastSAM -> instance masks.
  3. A mask is kept only from the tile whose centre is nearest to the mask
     centroid, so every object is owned by exactly one tile and no
     cross-tile merging is needed (the overlap is larger than an object).
  4. Filter the masks into "one object" masks:
       - background / shadow masks: must be clearly darker than the
         (median) background brightness,
       - tiny fragments and huge region masks: area relative to the median
         object area,
       - "group" masks that cover several other masks are dropped and their
         members kept; "part" masks (e.g. a screw head) that lie inside a
         kept mask are dropped.
  5. Count = number of remaining masks; each is painted in its own colour.

Usage:
  python count_ai.py --data ../../data --out outputs [--model FastSAM-s.pt]
"""
import argparse
import csv
import glob
import os
import pickle

import cv2
import numpy as np
from ultralytics import FastSAM

WORK_SIZE = 2400
TILE = 1024
OVERLAP = 256
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def tile_origins(length):
    if length <= TILE:
        return [0]
    stride = TILE - OVERLAP
    xs = list(range(0, length - TILE, stride)) + [length - TILE]
    return sorted(set(xs))


def run_tiles(model, img, conf):
    """Return all raw FastSAM masks as (tile_x0, tile_y0, x, y, bbox_mask)."""
    H, W = img.shape[:2]
    masks = []
    for y0 in tile_origins(H):
        for x0 in tile_origins(W):
            tile = img[y0:y0 + TILE, x0:x0 + TILE]
            r = model(tile, imgsz=TILE, conf=conf, iou=0.7, max_det=2000,
                      retina_masks=True, verbose=False, device="cpu")[0]
            if r.masks is None:
                continue
            for m in r.masks.data.cpu().numpy().astype(np.uint8):
                ys, xs = np.nonzero(m)
                if len(xs) == 0:
                    continue
                bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                masks.append((x0, y0, x0 + bx0, y0 + by0, m[by0:by1, bx0:bx1].copy()))
    return masks


def owned_masks(raw, shape):
    """Each object is owned by the tile whose centre is nearest to the mask
    centroid (per axis). This de-duplicates masks seen in two overlapping
    tiles without any merging; the overlap guarantees the owner sees the
    whole object."""
    H, W = shape
    xs_o, ys_o = np.array(tile_origins(W)), np.array(tile_origins(H))
    cxs = xs_o + min(TILE, W) / 2
    cys = ys_o + min(TILE, H) / 2
    out = []
    for tx, ty, x, y, m in raw:
        yy, xx = np.nonzero(m)
        mx, my = x + xx.mean(), y + yy.mean()
        if xs_o[np.argmin(np.abs(cxs - mx))] == tx and ys_o[np.argmin(np.abs(cys - my))] == ty:
            out.append((x, y, m))
    return out


def to_full(mask_entry, shape):
    x, y, m = mask_entry
    full = np.zeros(shape, np.uint8)
    full[y:y + m.shape[0], x:x + m.shape[1]] = m
    return full


def filter_masks(masks, gray):
    if not masks:
        return []
    bg = float(np.median(gray))
    areas = np.array([m[2].sum() for m in masks], np.float64)
    means = np.array([gray[y:y + m.shape[0], x:x + m.shape[1]][m > 0].mean() for x, y, m in masks])
    keep = np.where((means < 0.80 * bg) & (areas >= 20))[0]
    if len(keep) == 0:
        return []
    med = np.median(areas[keep])
    keep = [i for i in keep if 0.15 * med <= areas[i] <= 15 * med]

    order = sorted(keep, key=lambda i: areas[i])  # small first

    # Containment graph: i -> bigger masks j that cover >70% of i.
    small_first = order
    contained_in = {i: [] for i in keep}
    boxes = {i: (masks[i][0], masks[i][1], masks[i][0] + masks[i][2].shape[1], masks[i][1] + masks[i][2].shape[0])
             for i in keep}
    for a_idx, i in enumerate(small_first):
        xi0, yi0, xi1, yi1 = boxes[i]
        for j in small_first[a_idx + 1:]:  # j is at least as large as i
            xj0, yj0, xj1, yj1 = boxes[j]
            if xi0 >= xj1 or xj0 >= xi1 or yi0 >= yj1 or yj0 >= yi1:
                continue
            ox0, oy0, ox1, oy1 = max(xi0, xj0), max(yi0, yj0), min(xi1, xj1), min(yi1, yj1)
            mi = masks[i][2][oy0 - yi0:oy1 - yi0, ox0 - xi0:ox1 - xi0]
            mj = masks[j][2][oy0 - yj0:oy1 - yj0, ox0 - xj0:ox1 - xj0]
            inter = float((mi & mj).sum())
            if inter > 0.7 * areas[i]:
                contained_in[i].append(j)

    # FastSAM returns a hierarchy: whole objects, their parts (screw head /
    # shank) and sometimes groups of touching objects. Start from top-level
    # masks; the typical top-level area is the size of one object. A
    # top-level mask much bigger than that, which has >=2 sub-masks, is a
    # group -> replace it by its (maximal) sub-masks, recursively. Otherwise
    # the mask is one object and its sub-masks are parts -> drop them.
    children = {i: [] for i in keep}
    for i, parents in contained_in.items():
        for j in parents:
            children[j].append(i)
    top = [i for i in keep if not contained_in[i]]

    def resolve(i, typical, depth=0):
        kids = children[i]
        maximal = [c for c in kids if not any(p in kids for p in contained_in[c])]
        whole = [c for c in maximal if areas[c] >= 0.5 * typical]  # objects, not parts
        if depth < 4 and areas[i] > 1.4 * typical and len(whole) >= 2:
            out = []
            for c in whole:
                out.extend(resolve(c, typical, depth + 1))
            return out
        return [i]

    def select(typical):
        res = []
        for i in top:
            res.extend(resolve(i, typical))
        return list(dict.fromkeys(res))

    # pass 1: typical size = median top-level mask; pass 2: typical size =
    # most common size among the selected masks (= one object), re-resolve.
    result = select(float(np.median([areas[i] for i in top])))
    typical = robust_mode([areas[i] for i in result])
    result = select(typical)

    # Touching objects that FastSAM returned only as one merged mask (no
    # sub-masks to fall back to): split by area into k = round(area/typical)
    # instances (k-means on pixel coordinates).
    out = []
    for i in result:
        x, y, m = masks[i]
        k = int(round(areas[i] / typical)) if areas[i] >= 1.5 * typical else 1
        if k <= 1:
            out.append((x, y, m))
            continue
        ys, xs = np.nonzero(m)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, lab, _ = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        for j in range(k):
            sub = np.zeros_like(m)
            sub[ys[lab[:, 0] == j], xs[lab[:, 0] == j]] = 1
            out.append((x, y, sub))
    return out


def robust_mode(values):
    """Iterated median of the values near the estimate -> most common size."""
    a = np.asarray(values, np.float64)
    est = float(np.percentile(a, 25))
    for _ in range(30):
        sel = a[(a > 0.6 * est) & (a < 1.4 * est)]
        if len(sel) == 0:
            break
        new = float(np.median(sel))
        if abs(new - est) < 1e-3:
            break
        est = new
    return est


def overlay(img, masks, count):
    out = img.copy()
    rng = np.random.default_rng(0)
    for x, y, m in masks:
        col = rng.integers(40, 255, size=3).astype(np.float32)
        roi = out[y:y + m.shape[0], x:x + m.shape[1]]
        sel = m > 0
        roi[sel] = (0.4 * roi[sel] + 0.6 * col).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, cnts, -1, (255, 255, 255), 1)
    sc = max(img.shape[:2]) / 1600
    cv2.rectangle(out, (0, 0), (int(440 * sc), int(70 * sc)), (0, 0, 0), -1)
    cv2.putText(out, f"Count: {count}", (int(10 * sc), int(52 * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                1.5 * sc, (0, 255, 255), max(2, int(3 * sc)))
    return out


def count_image(model, img, conf=0.2, cache=None):
    s = min(1.0, WORK_SIZE / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if cache and os.path.exists(cache):
        with open(cache, "rb") as fh:
            raw = pickle.load(fh)
    else:
        raw = run_tiles(model(), small, conf)
        if cache:
            with open(cache, "wb") as fh:
                pickle.dump(raw, fh)
    return small, filter_masks(owned_masks(raw, gray.shape), gray)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--model", default="FastSAM-s.pt")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--cache", default="outputs/cache",
                    help="folder for raw FastSAM masks (re-filtering without re-running the model)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    _model = []

    def model():  # lazy: not needed when everything is cached
        if not _model:
            _model.append(FastSAM(args.model))
        return _model[0]

    files = sorted(f for f in glob.glob(os.path.join(args.data, "**", "*"), recursive=True)
                   if f.lower().endswith(IMG_EXT))
    rows = []
    for f in files:
        rel = os.path.relpath(f, args.data).replace("\\", "/")
        cache = os.path.join(args.cache, rel.replace("/", "__") + f"_{args.conf}.pkl")
        small, masks = count_image(model, cv2.imread(f), args.conf, cache)
        cv2.imwrite(os.path.join(args.out, rel.replace("/", "__")), overlay(small, masks, len(masks)))
        rows.append((rel, len(masks)))
        print(f"{rel:50s} {len(masks)}", flush=True)
    with open(os.path.join(args.out, "counts.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "count"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
