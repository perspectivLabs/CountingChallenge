"""
Count objects + overlay instance masks with a trained YOLO-seg model.

Tiled inference: the image is cut into overlapping 640px tiles; each detection is kept only
by the tile that "owns" its box centre (ownership boundaries sit at the middle of every
overlap), so objects are counted exactly once without any cross-tile merging.

  python count_ai.py --weights best.pt --input <images_dir> --output results [--gt ../ground_truth.csv] [--test-only]
"""
import argparse, csv, glob, os, time
import cv2
import numpy as np
from ultralytics import YOLO
from common import load, overlay, TILE, OVERLAP, TEST_IMAGES

def origins(n, tile=TILE, overlap=OVERLAP):
    step = tile - overlap
    o = list(range(0, max(n - tile, 0) + 1, step)) or [0]
    if o[-1] + tile < n: o.append(n - tile)
    return [max(v, 0) for v in o]

def owner_bounds(o, n, tile=TILE):
    """Half-open [lo, hi) range of coordinates each tile owns."""
    b = []
    for i, s in enumerate(o):
        lo = 0 if i == 0 else (s + o[i-1] + tile) / 2      # middle of overlap with previous tile
        hi = n if i == len(o) - 1 else (o[i+1] + s + tile) / 2
        b.append((lo, hi))
    return b

def predict(model, im, conf=0.25, iou=0.6):
    """Tiled inference. Returns a list of detections (score, y0, x0, mask_crop),
    with each mask cropped to its bounding box to keep memory small."""
    H, W = im.shape[:2]
    ys, xs = origins(H), origins(W)
    yb, xb = owner_bounds(ys, H), owner_bounds(xs, W)
    dets = []
    for iy, y0 in enumerate(ys):
        for ix, x0 in enumerate(xs):
            t = im[y0:y0+TILE, x0:x0+TILE]
            r = model.predict(t, imgsz=TILE, conf=conf, iou=iou, retina_masks=True,
                              max_det=1000, verbose=False)[0]
            if r.masks is None: continue
            md = r.masks.data.cpu().numpy() > 0.5
            bx = r.boxes.xyxy.cpu().numpy(); sc = r.boxes.conf.cpu().numpy()
            th, tw = t.shape[:2]
            for m, (x1, y1, x2, y2), s in zip(md, bx, sc):
                cy, cx = y0 + (y1 + y2) / 2, x0 + (x1 + x2) / 2
                if not (yb[iy][0] <= cy < yb[iy][1] and xb[ix][0] <= cx < xb[ix][1]):
                    continue
                m = m[:th, :tw]
                yy, xx = np.nonzero(m)
                if len(yy) == 0: continue
                a0, a1, b0, b1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
                dets.append((float(s), y0 + a0, x0 + b0, m[a0:a1, b0:b1].copy()))
    return dets

def _overlap(d1, d2):
    """Intersection / smaller-area for two cropped masks."""
    _, y1, x1, m1 = d1; _, y2, x2, m2 = d2
    ya, yb_ = max(y1, y2), min(y1 + m1.shape[0], y2 + m2.shape[0])
    xa, xb_ = max(x1, x2), min(x1 + m1.shape[1], x2 + m2.shape[1])
    if ya >= yb_ or xa >= xb_: return 0.0
    inter = np.logical_and(m1[ya-y1:yb_-y1, xa-x1:xb_-x1], m2[ya-y2:yb_-y2, xa-x2:xb_-x2]).sum()
    return inter / max(1, min(m1.sum(), m2.sum()))

def postprocess(dets, conf, dup_thr=0.5, min_rel_area=0.2):
    """Confidence filter, mask-level duplicate removal, and removal of tiny fragments."""
    d = sorted([x for x in dets if x[0] >= conf], key=lambda x: -x[0])
    kept = []
    for x in d:
        if all(_overlap(x, k) < dup_thr for k in kept):
            kept.append(x)
    if kept:
        med = np.median([k[3].sum() for k in kept])
        kept = [k for k in kept if k[3].sum() >= min_rel_area * med]
    return kept

def to_full(dets, shape):
    out = []
    for _, y, x, m in dets:
        f = np.zeros(shape, bool); f[y:y+m.shape[0], x:x+m.shape[1]] = m; out.append(f)
    return out

def load_gt(path):
    gt = {}
    if path and os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r.get("gt_count", "").strip(): gt[r["image"].strip()] = int(r["gt_count"])
    return gt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="results")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--conf", type=float, default=0.10, help="confidence threshold (0.10 chosen by --sweep on training images)")
    ap.add_argument("--dup", type=float, default=0.5,
                    help="two masks overlapping more than this (of the smaller) count as one object")
    ap.add_argument("--test-only", action="store_true", help="only the held-out test images")
    ap.add_argument("--sweep", action="store_true",
                    help="pick --conf on TRAINING images only (test images untouched)")
    a = ap.parse_args()
    model = YOLO(a.weights)
    gt = load_gt(a.gt)
    paths = sorted(glob.glob(os.path.join(a.input, "**", "*.jpg"), recursive=True))

    if a.sweep:
        paths = [p for p in paths if os.path.basename(p) not in TEST_IMAGES and os.path.basename(p) in gt]
        confs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        dups = [0.5, 0.6, 0.7, 0.8, 0.9]
        grid = {(c, d): [] for c in confs for d in dups}
        for p in paths:
            name = os.path.basename(p); t = time.time()
            dets = predict(model, load(p), conf=min(confs)); g = gt[name]
            for c in confs:
                for d in dups:
                    n = len(postprocess(dets, c, dup_thr=d)); grid[(c, d)].append(max(0.0, 1 - abs(n - g) / g))
            print(f"{name:30s} done ({time.time()-t:.0f}s)", flush=True)
        print("\nMean accuracy on TRAINING images  (rows: conf, columns: duplicate threshold)")
        print("conf  " + "".join(f"  dup={d:.1f}" for d in dups))
        for c in confs:
            print(f"{c:.2f}  " + "".join(f"  {100*np.mean(grid[(c, d)]):6.2f}%" for d in dups))
        best = max(grid, key=lambda k: np.mean(grid[k]))
        print(f"\nBest: conf={best[0]:.2f} dup={best[1]:.1f}  ->  run with --test-only --conf {best[0]:.2f} --dup {best[1]:.1f}")
        return

    os.makedirs(os.path.join(a.output, "overlays"), exist_ok=True)
    if a.test_only: paths = [p for p in paths if os.path.basename(p) in TEST_IMAGES]
    rows, accs = [], []
    for p in paths:
        t = time.time(); name = os.path.basename(p)
        im = load(p)
        dets = postprocess(predict(model, im, conf=a.conf), a.conf, dup_thr=a.dup); n = len(dets)
        vis = overlay(im, to_full(dets, im.shape[:2]), f"Count: {n}")
        vis = cv2.resize(vis, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(a.output, "overlays", name), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        g = gt.get(name); acc = None if g is None else max(0.0, 1 - abs(n - g) / g)
        if acc is not None: accs.append(acc)
        rows.append(dict(image=name, split="test" if name in TEST_IMAGES else "train",
                         predicted=n, gt_count="" if g is None else g,
                         accuracy="" if acc is None else f"{100*acc:.2f}", seconds=f"{time.time()-t:.1f}"))
        print(f"{name:30s} pred={n:4d}" + ("" if g is None else f"  gt={g:4d}  acc={100*acc:6.2f}%"), flush=True)
    with open(os.path.join(a.output, "counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if accs: print(f"\nMean count accuracy over {len(accs)} images with GT: {100*np.mean(accs):.2f}%")

if __name__ == "__main__":
    main()
