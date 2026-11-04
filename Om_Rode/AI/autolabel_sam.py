"""Auto-label the dataset with SAM 2.1 (instance masks) for YOLOv8-seg training.

Minimal-human-in-the-loop labelling:
  1. Images are rescaled (12 MP photos -> long side 2400 px) and cut into overlapping
     1024 px tiles. SAM 2.1 runs in "segment everything" mode on each tile.
  2. Masks touching an interior tile edge are discarded (the object is fully inside a
     neighbouring tile thanks to the overlap). Duplicates across tiles are merged by IoU.
  3. Masks are filtered with a classical foreground mask (dark objects on light
     background): holes / background / shadows are removed. "Group" masks that cover
     several objects and "part" masks that lie inside one object are resolved.
  4. Recovery pass: foreground blobs not covered by any mask are re-segmented with a
     SAM point prompt.
  5. Outputs: YOLO-seg polygon labels, a numbered overlay per image for review,
     a compressed mask file and counts.csv.

Usage:
  python autolabel_sam.py --data ../../data --out labels_sam --weights sam2.1_b.pt
"""
import argparse
import csv
import glob
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from ultralytics import SAM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Non_AI"))
from count_objects import foreground_mask, single_object_area  # noqa: E402

TILE, STRIDE, MAX_SIDE = 1024, 768, 2400


def tiles(h, w):
    ys = list(range(0, max(h - TILE, 0) + 1, STRIDE)) or [0]
    xs = list(range(0, max(w - TILE, 0) + 1, STRIDE)) or [0]
    if ys[-1] + TILE < h:
        ys.append(h - TILE)
    if xs[-1] + TILE < w:
        xs.append(w - TILE)
    return [(y, x) for y in ys for x in xs]


def iou_dedupe(masks, scores, thr=0.5):
    """masks: list of (y0,x0,crop bool). Greedy NMS by mask IoU."""
    order = np.argsort(-np.asarray(scores))
    kept = []
    canvas = {}
    for i in order:
        y0, x0, m = masks[i]
        dup = False
        for j in kept:
            yj, xj, mj = masks[j]
            # bbox overlap check
            ya, xa = max(y0, yj), max(x0, xj)
            yb, xb = min(y0 + m.shape[0], yj + mj.shape[0]), min(x0 + m.shape[1], xj + mj.shape[1])
            if ya >= yb or xa >= xb:
                continue
            a = m[ya - y0:yb - y0, xa - x0:xb - x0]
            b = mj[ya - yj:yb - yj, xa - xj:xb - xj]
            inter = np.logical_and(a, b).sum()
            if inter / (m.sum() + mj.sum() - inter) > thr:
                dup = True
                break
        if not dup:
            kept.append(i)
    return [masks[i] for i in kept], [scores[i] for i in kept]


# ---- masks are kept as crops: (y0, x0, bool array) -- never as full-image arrays ----
def crop(m):
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return (int(y0), int(x0), m[y0:y1, x0:x1].copy())


def overlap(a, b, pad=0):
    """Intersection window of two crops (optionally padded); None if disjoint."""
    ya, xa, ma = a
    yb, xb, mb = b
    y0, x0 = max(ya, yb) - pad, max(xa, xb) - pad
    y1, x1 = min(ya + ma.shape[0], yb + mb.shape[0]) + pad, min(xa + ma.shape[1], xb + mb.shape[1]) + pad
    if y0 >= y1 or x0 >= x1:
        return None
    return y0, x0, y1, x1


def window(c, y0, x0, y1, x1):
    """Crop c rendered into the window [y0:y1, x0:x1] (zero padded)."""
    yc, xc, m = c
    out = np.zeros((y1 - y0, x1 - x0), bool)
    sy0, sx0 = max(y0, yc), max(x0, xc)
    sy1, sx1 = min(y1, yc + m.shape[0]), min(x1, xc + m.shape[1])
    if sy0 < sy1 and sx0 < sx1:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = m[sy0 - yc:sy1 - yc, sx0 - xc:sx1 - xc]
    return out


def inter_area(a, b):
    w = overlap(a, b)
    return 0 if w is None else int(np.logical_and(window(a, *w), window(b, *w)).sum())


def union(cs):
    y0 = min(c[0] for c in cs)
    x0 = min(c[1] for c in cs)
    y1 = max(c[0] + c[2].shape[0] for c in cs)
    x1 = max(c[1] + c[2].shape[1] for c in cs)
    out = np.zeros((y1 - y0, x1 - x0), bool)
    for c in cs:
        out |= window(c, y0, x0, y1, x1)
    return (y0, x0, out)


def area(c):
    return int(c[2].sum())


def resolve_groups_and_parts(masks):
    """Remove masks covering several objects (groups) and masks that are part of one."""
    n = len(masks)
    areas = np.array([area(c) for c in masks])
    keep = np.ones(n, bool)
    for i in np.argsort(-areas):  # big first
        if not keep[i]:
            continue
        inside = [j for j in range(n) if j != i and keep[j] and areas[j] < areas[i]
                  and overlap(masks[i], masks[j]) is not None
                  and inter_area(masks[i], masks[j]) > 0.8 * areas[j]]
        if not inside:
            continue
        cov = inter_area(masks[i], union([masks[j] for j in inside])) / areas[i]
        big_inside = [j for j in inside if areas[j] > 0.3 * areas[i]]
        if len(big_inside) >= 2 or (len(inside) >= 2 and cov > 0.6):
            keep[i] = False  # group mask
        else:
            for j in inside:
                keep[j] = False  # parts of i
    return [c for c, k in zip(masks, keep) if k]


def sam_single_area(masks, fg):
    """Area of one whole object, measured in SAM-mask units.

    Isolated foreground blobs whose area matches the single-object mode hold exactly one
    object; the union of the SAM masks over such a blob is that object's full extent
    (SAM includes shiny heads that the classical threshold partly misses)."""
    nb, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    if nb <= 1 or not masks:
        return 0.0
    a = st[1:, cv2.CC_STAT_AREA]
    single_fg = single_object_area(a[a > 100])
    out = []
    for i in range(1, nb):
        if not 0.75 * single_fg < st[i, cv2.CC_STAT_AREA] < 1.3 * single_fg:
            continue
        x, y, w, h = st[i, :4]
        blob = (int(y), int(x), lab[y:y + h, x:x + w] == i)
        over = [c for c in masks if overlap(blob, c) is not None and inter_area(blob, c) > 0.5 * area(c)]
        if over:
            out.append(area(union(over)))
    return float(np.median(out)) if out else single_fg


def merge_parts(masks, single, lo=0.55, hi=1.35, part=0.8, tiny_frac=0.3, tiny_hi=1.6):
    """Merge touching masks that are *parts* of one object (e.g. a shiny screw head and
    its dark shank). Two masks merge when each is clearly smaller than one object and
    their union is about one object. Two whole objects that cross are each ~1x the
    single area, so they are never merged. A tiny fragment (< tiny_frac) is absorbed by
    the neighbour it touches most, if the result is still a plausible size."""
    if single <= 0 or len(masks) < 2:
        return masks
    masks = list(masks)
    k = np.ones((7, 7), np.uint8)
    while True:
        areas = [area(c) for c in masks]
        best = None
        for i in range(len(masks)):
            if areas[i] >= part * single:
                continue
            tiny = areas[i] < tiny_frac * single
            for j in range(len(masks)):
                if j == i or (areas[j] >= part * single and not tiny):
                    continue
                u = areas[i] + areas[j]
                if not (lo * single < u < (tiny_hi if tiny else hi) * single):
                    continue
                w = overlap(masks[i], masks[j], pad=5)
                if w is None:
                    continue
                di = cv2.dilate(window(masks[i], *w).astype(np.uint8), k)
                contact = int(np.logical_and(di > 0, window(masks[j], *w)).sum())
                if contact > 10 and (best is None or contact > best[0]):
                    best = (contact, i, j)
        if not best:
            return masks
        _, i, j = best
        masks[i] = union([masks[i], masks[j]])
        masks.pop(j)


def save_masks(path, masks, shape):
    boxes = np.array([(c[0], c[1], *c[2].shape) for c in masks], np.int32).reshape(-1, 4)
    bits = [np.packbits(c[2].ravel()) for c in masks]
    offs = np.cumsum([0] + [len(b) for b in bits])
    np.savez_compressed(path, boxes=boxes, bits=np.concatenate(bits) if bits else np.zeros(0, np.uint8),
                        offsets=offs, shape=np.array(shape))


def load_masks(path):
    """Returns (list of (y0, x0, crop), (H, W))."""
    d = np.load(path)
    out = []
    for k, (y0, x0, h, w) in enumerate(d["boxes"]):
        b = d["bits"][d["offsets"][k]:d["offsets"][k + 1]]
        out.append((int(y0), int(x0), np.unpackbits(b)[:h * w].reshape(h, w).astype(bool)))
    return out, tuple(d["shape"])


def sam_prompt(model, t, prompt):
    r = model(t, verbose=False, **prompt)[0]
    if r.masks is None or len(r.masks.data) == 0:
        return None
    m = r.masks.data[0].cpu().numpy()[:t.shape[0], :t.shape[1]] > 0.5
    return m if m.any() else None


def distance_peaks(comp, min_sep):
    """Local maxima of the distance transform, at least min_sep apart (one per object)."""
    d = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    k = max(3, int(min_sep) | 1)
    mx = cv2.dilate(d, np.ones((k, k), np.uint8))
    ys, xs = np.where((d == mx) & (d > 0.5 * d.max()))
    pts = sorted(zip(ys, xs), key=lambda p: -d[p])
    out = []
    for p in pts:
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep ** 2 for q in out):
            out.append(p)
    return out


def add_new(mt, cy, cx, tshape, masks, covered, fg, med):
    """Add the not-yet-covered part of prompted mask mt (tile coords) as a new object."""
    cov = covered[cy:cy + tshape[0], cx:cx + tshape[1]]
    if (mt & cov).sum() > 0.4 * mt.sum():  # mostly an object we already have
        return 0
    new = mt & ~cov
    nn, nl, ns, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8), 8)
    if nn <= 1:
        return 0
    new = nl == 1 + int(np.argmax(ns[1:, cv2.CC_STAT_AREA]))
    if new.sum() < 0.4 * med:
        return 0
    if fg[cy:cy + tshape[0], cx:cx + tshape[1]][new].mean() < 0.5:
        return 0
    y0, x0, mc = crop(new)
    masks.append((y0 + cy, x0 + cx, mc))
    covered[y0 + cy:y0 + cy + mc.shape[0], x0 + cx:x0 + cx + mc.shape[1]] |= mc
    return 1


def autolabel(path, model):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]

    # classical foreground (computed at 1600 px, upsampled)
    s2 = 1600 / max(H, W)
    small = cv2.resize(img, (int(W * s2), int(H * s2)), interpolation=cv2.INTER_AREA) if s2 < 1 else img
    fg, norm, valid = foreground_mask(small)
    fg = cv2.resize(fg, (W, H), interpolation=cv2.INTER_NEAREST) > 0  # hole-filled
    valid = cv2.resize(valid, (W, H), interpolation=cv2.INTER_NEAREST) > 0
    # raw darkness (no hole filling): enclosed background gaps between parts are NOT dark
    t_otsu, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.resize((norm < t_otsu).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR) > 0

    masks, scores = [], []
    for (y, x) in tiles(H, W):
        t = img[y:y + TILE, x:x + TILE]
        r = model(t, verbose=False)[0]
        torch.cuda.empty_cache()
        if r.masks is None or len(r.masks.data) == 0:
            continue
        md = r.masks.data.cpu().numpy() > 0.5
        conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.ones(len(md))
        th, tw = t.shape[:2]
        for m, c in zip(md, conf):
            m = m[:th, :tw]
            if m.sum() < 50:
                continue
            ys, xs = np.where(m)
            touches = ((ys.min() <= 1 and y > 0) or (xs.min() <= 1 and x > 0) or
                       (ys.max() >= th - 2 and y + th < H) or (xs.max() >= tw - 2 and x + tw < W))
            if touches:
                continue
            y0, x0, mc = crop(m)
            masks.append((y + y0, x + x0, mc))
            scores.append(float(c))

    # foreground filter: object masks are mostly dark foreground inside the valid view
    good = []
    for i, (y0, x0, m) in enumerate(masks):
        f = fg[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m].mean()
        v = valid[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m].mean()
        if f > 0.6 and v > 0.9:
            good.append(i)
    masks = [masks[i] for i in good]
    scores = [scores[i] for i in good]
    masks, scores = iou_dedupe(masks, scores)

    masks = resolve_groups_and_parts(masks)
    single = sam_single_area(masks, fg & valid)

    # size sanity: drop tiny fragments relative to the typical object
    areas = np.array([area(c) for c in masks])
    med = np.median(areas) if len(areas) else 0
    masks = [c for c, a in zip(masks, areas) if a > 0.2 * med]

    # recovery pass: SAM's automatic mode misses some objects (esp. black screws touching
    # others). Uncovered, *thick* foreground blobs are re-segmented with a point + box
    # prompt; only the newly covered pixels are kept. Repeated until nothing changes.
    covered = np.zeros((H, W), bool)
    for y0, x0, m in masks:
        covered[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] |= m
    added = 0
    for _ in range(3):
        rest = (dark & valid & ~covered).astype(np.uint8)
        rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(rest, 8)
        round_added = 0
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 0.25 * med:
                continue
            x, y, w, h = st[i, :4]
            comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
            d = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
            if d.max() < 0.15 * np.sqrt(med):  # thin sliver/shadow, not an object
                continue
            py, px = np.unravel_index(np.argmax(d), d.shape)
            py, px = py + y, px + x
            cy, cx = min(max(py - TILE // 2, 0), max(H - TILE, 0)), min(max(px - TILE // 2, 0), max(W - TILE, 0))
            t = img[cy:cy + TILE, cx:cx + TILE]
            box = [max(x - cx - 8, 0), max(y - cy - 8, 0), min(x + w - cx + 8, t.shape[1]), min(y + h - cy + 8, t.shape[0])]
            mts = []
            pt = [[int(px - cx), int(py - cy)]]
            for prompt in (dict(points=pt, labels=[1]), dict(bboxes=[box], points=pt, labels=[1])):
                cand = sam_prompt(model, t, prompt)
                if cand is not None and 0.25 * med < cand.sum() < 1.8 * med:  # about one object
                    mts = [cand]
                    break
            if not mts:
                # touching objects (e.g. two black screws): one prompt per distance peak,
                # the other peaks as negative points
                seeds = distance_peaks(comp, 0.5 * np.sqrt(single or med))
                if len(seeds) >= 2:
                    seeds = [(sy + y - cy, sx + x - cx) for sy, sx in seeds]
                    for k, (sy, sx) in enumerate(seeds):
                        others = [[int(ox), int(oy)] for j, (oy, ox) in enumerate(seeds) if j != k]
                        cand = sam_prompt(model, t, dict(points=[[[int(sx), int(sy)]] + others], labels=[[1] + [0] * len(others)]))
                        if cand is not None and 0.25 * med < cand.sum() < 1.8 * med:
                            mts.append(cand)
            for mt in mts:
                round_added += add_new(mt, cy, cx, t.shape, masks, covered, dark, med)
        added += round_added
        if round_added == 0:
            break
    # merge object parts (head/shank) incl. parts found by the recovery pass
    masks = merge_parts(masks, single)
    return img, masks, added


def mask_to_polygon(c):
    y0, x0, m = c
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x0, y0))
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True)
    return c.reshape(-1, 2)


def draw(img, masks):
    ov = img.copy()
    rng = np.random.default_rng(0)
    for y0, x0, m in masks:
        roi = ov[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        roi[m] = (0.45 * roi[m] + 0.55 * rng.integers(40, 255, 3)).astype(np.uint8)
    for k, (y0, x0, m) in enumerate(masks):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x0, y0))
        cv2.drawContours(ov, cnts, -1, (0, 0, 0), 1)
        ys, xs = np.where(m)
        cv2.putText(ov, str(k + 1), (int(xs.mean()) + x0 - 8, int(ys.mean()) + y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.rectangle(ov, (0, 0), (420, 60), (255, 255, 255), -1)
    cv2.putText(ov, f"SAM count: {len(masks)}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="Om_Rode/AI/labels_sam")
    ap.add_argument("--weights", default="weights/sam2.1_b.pt")
    ap.add_argument("--only", default="", help="substring filter on image names")
    args = ap.parse_args()
    for d in ("images", "labels", "overlays", "masks"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    model = SAM(args.weights)
    rows = []
    for p in sorted(glob.glob(os.path.join(args.data, "**", "*.jpg"), recursive=True)):
        if args.only and args.only not in p:
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        t0 = time.time()
        img, masks, n_added = autolabel(p, model)
        H, W = img.shape[:2]
        cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), img)
        with open(os.path.join(args.out, "labels", name + ".txt"), "w") as f:
            for c in masks:
                poly = mask_to_polygon(c) / np.array([W, H])
                f.write("0 " + " ".join(f"{v:.5f}" for v in poly.ravel()) + "\n")
        save_masks(os.path.join(args.out, "masks", name + ".npz"), masks, (H, W))
        cv2.imwrite(os.path.join(args.out, "overlays", name + ".jpg"), draw(img, masks))
        rows.append((name, len(masks), n_added))
        print(f"{name}: {len(masks)} (recovered {n_added}) [{time.time() - t0:.0f}s]", flush=True)
    with open(os.path.join(args.out, "counts.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("image", "sam_count", "recovered_by_prompt"), *rows])


if __name__ == "__main__":
    main()
