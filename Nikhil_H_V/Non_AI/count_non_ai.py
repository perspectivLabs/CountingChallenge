
import argparse
import csv
import glob
import os

import cv2
import numpy as np

WORK_SIZE = 2400
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ----------------------------------------------------------------- masks ---
def foreground_mask(gray):
    g = gray.astype(np.float32)
    ksz = int(round(gray.shape[1] / 40)) | 1
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
    bg = cv2.GaussianBlur(bg, (0, 0), ksz / 4)
    flat8 = (np.clip(g / (bg + 1.0), 0, 1) * 255).astype(np.uint8)
    flat8 = cv2.GaussianBlur(flat8, (3, 3), 0)
    _, m = cv2.threshold(flat8, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def robust_mode(values, start=None):
    """Iterated median of values near the estimate -> most common size."""
    a = np.asarray(values, np.float64)
    if len(a) == 0:
        return 0.0
    est = np.percentile(a, 25) if start is None else start
    for _ in range(30):
        sel = a[(a > 0.6 * est) & (a < 1.4 * est)]
        if len(sel) == 0:
            break
        new = float(np.median(sel))
        if abs(new - est) < 1e-3:
            break
        est = new
    return est


def find_holes(mask):
    """Return (hole_label_img, list of (label, area, cx, cy)) for nut holes."""
    inv = (1 - mask).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(inv, 4)
    H, W = mask.shape
    cands = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 15 or x == 0 or y == 0 or x + w == W or y + h == H or a > 0.01 * H * W:
            continue
        cnt, _ = cv2.findContours((lab[y:y + h, x:x + w] == i).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        p = cv2.arcLength(cnt[0], True)
        circ = 4 * np.pi * a / (p * p + 1e-6)
        cands.append((i, a, circ, cen[i][0], cen[i][1]))
    round_ones = [c for c in cands if c[2] > 0.8]
    if len(round_ones) < 3:
        return lab, [], cands
    med = robust_mode([c[1] for c in round_ones], start=np.median([c[1] for c in round_ones]))
    holes = [c for c in cands if 0.55 * med < c[1] < 1.8 * med and c[2] > 0.72]
    return lab, holes, cands


# ------------------------------------------------------------- splitting ---
def split_blob(blob, k, seeds=()):
    """Split a binary blob (uint8) into k labels using watershed."""
    if k <= 1:
        return blob.astype(np.int32)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    seeds = [tuple(map(int, s)) for s in seeds][:k]
    if len(seeds) < k:  # add distance-transform peaks far from existing seeds
        dil = cv2.dilate(dist, np.ones((5, 5), np.uint8))
        peaks = np.argwhere((dist == dil) & (dist > 0.4 * dist.max()))
        order = np.argsort(-dist[peaks[:, 0], peaks[:, 1]])
        min_d = max(3.0, 1.2 * np.median(dist[dist > 0]) * 2)
        for idx in order:
            y, x = peaks[idx]
            if all(np.hypot(y - sy, x - sx) >= min_d for sy, sx in seeds):
                seeds.append((int(y), int(x)))
            if len(seeds) == k:
                break
    if len(seeds) < k:  # fallback: k-means on pixel coordinates
        ys, xs = np.nonzero(blob)
        pts = np.column_stack([ys, xs]).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, _, centers = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        seeds = [tuple(np.round(c).astype(int)) for c in centers]
    markers = np.zeros(blob.shape, np.int32)
    markers[blob == 0] = k + 1
    for j, (y, x) in enumerate(seeds, 1):
        cv2.circle(markers, (int(x), int(y)), 1, j, -1)
    markers[blob == 0] = k + 1
    relief = cv2.cvtColor((255 * (1 - dist / (dist.max() + 1e-6))).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(relief, markers)
    markers[(markers == k + 1) | (blob == 0)] = 0
    # watershed lines (-1) inside the blob -> nearest neighbour label
    if (markers < 0).any():
        lines = (markers < 0) & (blob > 0)
        dil = cv2.dilate(np.maximum(markers, 0).astype(np.float32), np.ones((3, 3))).astype(np.int32)
        markers[lines] = dil[lines]
        markers[markers < 0] = 0
    return markers


# --------------------------------------------------------------- counting ---
def count_image(img):
    s = min(1.0, WORK_SIZE / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask = foreground_mask(gray)

    hole_lab, holes, cands = find_holes(mask)
    # fill every small enclosed region (nut holes and tiny gaps) -> solid blobs
    hole_med = np.median([h[1] for h in holes]) if holes else 0
    fill_ids = [c[0] for c in cands if c[1] <= max(1.5 * hole_med, 0.00002 * mask.size)]
    lut = np.zeros(hole_lab.max() + 1, np.uint8)
    lut[fill_ids] = 1
    filled = np.maximum(mask, lut[hole_lab])

    # Group fragments of one object (shiny heads break up into pieces) with a
    # closing, but measure area on the un-closed mask so that the gaps between
    # touching objects do not inflate cluster areas.
    c = max(5, int(round(max(gray.shape) / 250))) | 1
    grouped = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grouped, 8)
    if n <= 1:
        return small, 0, np.zeros(gray.shape, np.int32)
    closed_areas = st[:, 4].astype(np.float64)  # includes shiny gaps in heads
    lab[filled == 0] = 0
    areas = np.bincount(lab.ravel(), minlength=n).astype(np.float64)  # raw pixels
    big = areas[1:][areas[1:] > 30]
    typical = robust_mode(big, start=np.median(big))
    dust = 0.25 * typical

    holes_in = {i: [] for i in range(1, n)}
    for h in holes:
        li = lab[int(round(h[4])), int(round(h[3]))]
        if li > 0:
            holes_in[li].append(h)
    # Real nut holes: a nut is a ring, so blob area / hole area is small (~3-8).
    # Reflections inside screw/bolt heads give a much larger ratio -> ignore.
    one = [i for i in range(1, n) if len(holes_in[i]) == 1]
    ratios = [areas[i] / holes_in[i][0][1] for i in one]
    if len(one) < 5 or np.median(ratios) > 10:
        holes_in = {i: [] for i in range(1, n)}
        one = []
    nut_area = robust_mode([areas[i] for i in one]) if one else 0
    # Round solid parts (thick nuts / bushings seen without a visible hole):
    # isolated, round, solid blobs give the disc size; Hough circles of that
    # size then find discs everywhere, also inside clusters.
    discs_in = {i: 0 for i in range(1, n)}
    disc_single = set()
    for i in range(1, n):
        if holes_in[i] or areas[i] <= dust:
            continue
        x, y, w, h = st[i][:4]
        blob = (cv2.morphologyEx((lab[y:y + h, x:x + w] == i).astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((5, 5), np.uint8)))
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        a, p = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
        hull = cv2.contourArea(cv2.convexHull(cnt))
        if p > 0 and 4 * np.pi * a / (p * p) > 0.80 and a / (hull + 1e-6) > 0.93:
            disc_single.add(i)
    disc_area = disc_area_raw = 0.0
    if len(disc_single) >= 3:
        disc_area = robust_mode([closed_areas[i] for i in disc_single])
        disc_area_raw = robust_mode([areas[i] for i in disc_single])
        r = np.sqrt(disc_area / np.pi)
        circles = cv2.HoughCircles(cv2.GaussianBlur(gray, (5, 5), 1.5), cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=1.6 * r, param1=100, param2=22,
                                   minRadius=int(0.8 * r), maxRadius=int(1.2 * r) + 1)
        yy, xx = np.mgrid[-int(r):int(r) + 1, -int(r):int(r) + 1]
        disk = (xx ** 2 + yy ** 2) <= (0.9 * r) ** 2
        for cx, cy, _ in (circles[0] if circles is not None else []):
            cx, cy = int(round(cx)), int(round(cy))
            y0, x0 = cy - int(r), cx - int(r)
            if y0 < 0 or x0 < 0 or y0 + disk.shape[0] > filled.shape[0] or x0 + disk.shape[1] > filled.shape[1]:
                continue
            if filled[y0:y0 + disk.shape[0], x0:x0 + disk.shape[1]][disk].mean() < 0.9:
                continue  # circle not solid foreground -> not a disc
            li = lab[cy, cx]
            if li > 0 and not holes_in[li]:
                discs_in[li] += 1
        for i in disc_single:  # an isolated disc is one disc, whatever Hough says
            discs_in[i] = 1

    # screws / bolts: measured on the closed mask (shiny heads are filled in)
    no_hole = [closed_areas[i] for i in range(1, n)
               if not holes_in[i] and i not in disc_single and areas[i] > dust]
    part_area = robust_mode(no_hole) if len(no_hole) >= 3 else (nut_area or typical)
    part_area_raw = robust_mode([areas[i] for i in range(1, n)
                                 if not holes_in[i] and i not in disc_single and areas[i] > dust]) \
        if len(no_hole) >= 3 else part_area

    instances = np.zeros(gray.shape, np.int32)
    total = 0
    for i in range(1, n):
        x, y, w, h = st[i][:4]
        hs = holes_in.get(i, [])
        if not hs and (areas[i] < dust or closed_areas[i] < 0.3 * part_area):  # dust
            continue
        if hs:  # nuts (+ maybe screws touching them): raw area, nut holes filled
            rest = max(0.0, areas[i] - len(hs) * nut_area)
            k = len(hs) + int(round(rest / part_area_raw))
        else:  # screws / bolts (+ discs found by Hough)
            nd = discs_in[i]
            k = nd + int(round(max(0.0, areas[i] - nd * disc_area_raw) / part_area_raw))
        k = max(k, 1)
        pad = 2
        blob = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        blob = cv2.copyMakeBorder(blob, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        seeds = [(hh[4] - y + pad, hh[3] - x + pad) for hh in hs]
        parts = split_blob(blob, k, seeds)[pad:-pad, pad:-pad]
        region = instances[y:y + h, x:x + w]
        inside = lab[y:y + h, x:x + w] == i
        region[inside] = np.where(parts[inside] > 0, parts[inside] + total, total + 1)
        total += k
    return small, total, instances


# ---------------------------------------------------------------- drawing ---
def overlay(img, instances, count):
    rng = np.random.default_rng(0)
    colors = rng.integers(40, 255, size=(instances.max() + 1, 3)).astype(np.uint8)
    colors[0] = 0
    fg = instances > 0
    out = img.copy()
    out[fg] = (0.4 * img[fg] + 0.6 * colors[instances][fg]).astype(np.uint8)
    edges = cv2.morphologyEx(instances.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3))) > 0
    out[edges] = (255, 255, 255)
    sc = max(img.shape[:2]) / 1600
    cv2.rectangle(out, (0, 0), (int(440 * sc), int(70 * sc)), (0, 0, 0), -1)
    cv2.putText(out, f"Count: {count}", (int(10 * sc), int(52 * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                1.5 * sc, (0, 255, 255), max(2, int(3 * sc)))
    return out


def list_images(root):
    return sorted(f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                  if f.lower().endswith(IMG_EXT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for f in list_images(args.data):
        small, count, inst = count_image(cv2.imread(f))
        rel = os.path.relpath(f, args.data).replace("\\", "/")
        cv2.imwrite(os.path.join(args.out, rel.replace("/", "__")), overlay(small, inst, count))
        rows.append((rel, count))
        print(f"{rel:50s} {count}")
    with open(os.path.join(args.out, "counts.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "count"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
