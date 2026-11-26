"""Non-AI screw counter: classical OpenCV pipeline.

1. Downscale, estimate smooth background (large median/morph close) and take
   the difference -> illumination-invariant "foreground strength".
2. Otsu threshold + small morphological clean-up -> binary mask.
3. Connected components. Isolated blobs = 1 item. Merged/touching blobs are
   split with a distance-transform watershed; if the watershed is not
   confident, the count falls back to area / median single-item area.
4. Overlay coloured masks + numbered labels, write counts.csv.
"""
import argparse, csv, glob, os
import cv2, numpy as np

W = 1600  # working long side


def foreground(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    bg = cv2.GaussianBlur(bg, (0, 0), 25)
    d = cv2.subtract(bg, g)           # screws are darker than background
    d = cv2.GaussianBlur(d, (0, 0), 1.2)
    return d


def segment(img):
    d = foreground(img)
    t, _ = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(t * 0.6, 12)              # a bit permissive: thin shiny screws
    m = (d > t).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def split_and_count(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(30, np.percentile(areas, 90) * 0.04) if len(areas) else 30
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    ka = np.array([stats[i, cv2.CC_STAT_AREA] for i in keep], float)
    if len(ka) == 0:
        return np.zeros_like(lab), 0
    # single-item area: median of the smaller-than-median*1.4 blobs (robust to clumps)
    # robust single-item area: median of the smallest third of blobs (clumps are always larger)
    srt = np.sort(ka)
    single = np.median(srt[: max(1, len(srt) // 3)])
    out = np.zeros(mask.shape, np.int32); nid = 0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    for i in keep:
        comp = (lab == i)
        a = stats[i, cv2.CC_STAT_AREA]
        k = int(round(a / single))
        if k <= 1:
            nid += 1; out[comp] = nid; continue
        # clump: seeded watershed with k markers from k-means on pixel coords
        ys, xs = np.nonzero(comp)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        wts = dist[ys, xs]
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, lbl, _ = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lbl = lbl.ravel()
        for j in range(k):
            nid += 1
            out[ys[lbl == j], xs[lbl == j]] = nid
    return out, nid


def overlay(img, inst, n):
    rng = np.random.RandomState(1)
    col = rng.randint(60, 255, (n + 1, 3)).astype(np.uint8); col[0] = 0
    cm = col[inst]
    vis = img.copy()
    sel = inst > 0
    vis[sel] = (0.45 * img[sel] + 0.55 * cm[sel]).astype(np.uint8)
    cnts = []
    for i in range(1, n + 1):
        c, _ = cv2.findContours((inst == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts += c
    cv2.drawContours(vis, cnts, -1, (0, 0, 255), 1)
    cv2.putText(vis, f"count: {n}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="output")
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.input, "**", "*.jp*g"), recursive=True))
    rows = []
    for f in files:
        img = cv2.imread(f)
        s = W / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        inst, n = split_and_count(segment(img))
        name = os.path.splitext(os.path.relpath(f, a.input).replace(os.sep, "_"))[0]
        cv2.imwrite(os.path.join(a.output, name + "_overlay.jpg"), overlay(img, inst, n))
        rows.append((os.path.relpath(f, a.input), n)); print(rows[-1])
    with open(os.path.join(a.output, "counts.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "count"]); w.writerows(rows)

if __name__ == "__main__":
    main()
