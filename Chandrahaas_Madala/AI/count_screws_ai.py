"""AI screw counter: pretrained FastSAM (class-agnostic segmentation, zero-shot, no training).

FastSAM proposes masks for everything; we keep only masks that are
(a) small relative to the image, (b) darker than the local background
(rejects the table/background mask and shadows), then remove duplicates
(NMS on mask IoU). Count = number of surviving instance masks.
"""
import argparse, csv, glob, os
import cv2, numpy as np
from ultralytics import FastSAM


def local_dark(gray, m):
    ring = cv2.dilate(m, np.ones((15, 15), np.uint8)) & ~m
    if ring.sum() == 0 or m.sum() == 0:
        return 0.0
    return float(gray[ring > 0].mean() - gray[m > 0].mean())


def tile_masks(model, img, imgsz, conf, iou, min_contrast, max_frac):
    h, w = img.shape[:2]
    r = model(img, imgsz=imgsz, conf=conf, iou=iou, retina_masks=True, max_det=1500, verbose=False)[0]
    if r.masks is None:
        return []
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1)
    out = []
    for mk in r.masks.data.cpu().numpy():
        m = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        a = int(m.sum())
        if a < 40 or a > max_frac * 1600 * 1600:
            continue
        if local_dark(gray, m) < min_contrast:
            continue
        out.append(m)
    return out


def run(model, img, imgsz=1024, conf=0.1, iou=0.6, min_contrast=18, max_frac=0.0035, tile=800, stride=640):
    """Tile the image (small screws are tiny in a full frame), segment each tile,
    paste masks back, then de-duplicate masks seen in two overlapping tiles."""
    H, W = img.shape[:2]
    cand = []
    ys = list(range(0, max(1, H - tile), stride)) + [max(0, H - tile)]
    xs = list(range(0, max(1, W - tile), stride)) + [max(0, W - tile)]
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            t = img[y0:y0 + tile, x0:x0 + tile]
            for m in tile_masks(model, t, imgsz, conf, iou, min_contrast, max_frac):
                # drop masks touching the tile border (they are cut objects; the neighbouring tile sees them whole)
                edge = (y0 > 0 and m[0].any()) or (x0 > 0 and m[:, 0].any()) or \
                       (y0 + tile < H and m[-1].any()) or (x0 + tile < W and m[:, -1].any())
                if edge:
                    continue
                full = np.zeros((H, W), np.uint8)
                full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
                cand.append(full)
    cand.sort(key=lambda m: -int(m.sum()))
    out = []; cover = np.zeros((H, W), np.uint8)
    for m in cand:
        a = int(m.sum())
        if int((m & cover).sum()) > 0.4 * a:
            continue
        out.append(m); cover |= m
    return out


def corrected_count(masks):
    """FastSAM merges touching items into one mask. Estimate the single-item area
    from the smallest third of masks and count a mask of k x that area as k items."""
    if not masks:
        return 0
    a = np.sort([int(m.sum()) for m in masks])
    a = a[a >= 0.4 * np.median(a)]                 # ignore fragments
    single = np.median(a[: max(2, len(a) // 3)])
    return int(sum(min(4, max(1, round(x / single))) for x in a))


def overlay(img, masks, n=None):
    rng = np.random.RandomState(1); vis = img.copy()
    for m in masks:
        c = rng.randint(60, 255, 3)
        vis[m > 0] = (0.45 * vis[m > 0] + 0.55 * c).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (0, 0, 255), 1)
    cv2.putText(vis, f"count: {n if n is not None else len(masks)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", default="output")
    ap.add_argument("--weights", default="FastSAM-s.pt"); ap.add_argument("--long_side", type=int, default=1600)
    a = ap.parse_args(); os.makedirs(a.output, exist_ok=True)
    model = FastSAM(a.weights); rows = []
    for f in sorted(glob.glob(os.path.join(a.input, "**", "*.jp*g"), recursive=True)):
        img = cv2.imread(f); s = a.long_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        masks = run(model, img); n = corrected_count(masks)
        name = os.path.splitext(os.path.relpath(f, a.input).replace(os.sep, "_"))[0]
        cv2.imwrite(os.path.join(a.output, name + "_overlay.jpg"), overlay(img, masks, n))
        rows.append((os.path.relpath(f, a.input), n, len(masks))); print(rows[-1], flush=True)
    with open(os.path.join(a.output, "counts.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "count", "raw_masks"]); w.writerows(rows)

if __name__ == "__main__":
    main()
