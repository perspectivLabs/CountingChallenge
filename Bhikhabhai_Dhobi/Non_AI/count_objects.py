"""Classical (non-AI) object counter for the CountingChallenge dataset.

Pipeline (per image):
  1. Resize to a fixed working size and flatten uneven lighting on the LAB
     lightness channel (divide by a smooth background estimate).
  2. Otsu threshold -> foreground mask of dark objects. Dust specks and the
     dark lens-vignette corners (long contact with the frame) are removed.
  3. Nut holes: enclosed bright regions that are hole-sized and round
     (circularity >= 0.8; triangular gaps between touching nuts are < 0.7).
     Hole mode is used only if an image has >= 5 consistent holes, so a few
     gaps enclosed between touching screws are not mistaken for nuts.
  4. Nut blobs: one cell per hole (nearest-hole split), plus extra seeds when
     a blob is clearly bigger than its holes explain (a hole hidden by a
     neighbour).
  5. Other blobs: the area of one loose object is estimated per image; a blob
     holds k = round(area / unit) objects. k well-separated distance peaks
     are seeds, and a marker watershed whose landscape mixes the distance map
     with image brightness cuts touching parts along their bright rims.
  6. One region per object -> count, coloured mask overlay, counts.csv.

Usage:
  python count_objects.py --data ../../data --out output
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

WORK_SIDE = 2000  # long side of the working image in pixels
EDGE_WEIGHT = 0.5  # share of image brightness in the watershed landscape
SEED_SEPARATION = (0.7, 0.5, 0.35, 0.2)  # seed spacing tried, x sqrt(unit area)
HOLE_MIN_CIRCULARITY = 0.8  # holes measured >0.9, inter-nut gaps <0.7


def load_and_flatten(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = WORK_SIDE / max(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    # background: downsample, close away the dark objects, blur, upsample
    small = cv2.resize(L, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    small = cv2.GaussianBlur(cv2.morphologyEx(small, cv2.MORPH_CLOSE, k), (0, 0), 8)
    bg = cv2.resize(small, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_CUBIC)
    flat = np.clip(L / (bg + 1e-3), 0, 1.5)
    return img, np.clip(flat * 170, 0, 255).astype(np.uint8)


def foreground(flat_u8):
    _, m = cv2.threshold(flat_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 60  # drop dust
    # drop dark lens-vignette corners: they run a long way along the border,
    # while a real object cut by the frame only touches it briefly
    edge = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    contact = np.bincount(edge, minlength=n)
    keep[contact > 0.04 * max(m.shape)] = False
    return (keep[lab] * 255).astype(np.uint8)


def nut_holes(mask):
    """Enclosed, round, bright regions inside the foreground = nut holes."""
    inv = cv2.bitwise_not(mask)
    n, lab, st, cen = cv2.connectedComponentsWithStats(inv, connectivity=4)
    H, W = mask.shape
    holes = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue  # background touching the border
        if not (40 <= a <= 1500):
            continue
        if max(w, h) > 2.0 * min(w, h):
            continue
        fill = a / (w * h)  # a disc fills ~0.785 of its box
        if fill < 0.55:
            continue
        # roundness: a nut hole is a disc (~0.85+ after pixelisation), while
        # the gap enclosed between three touching nuts is a curved triangle
        cnts, _ = cv2.findContours((lab[y:y + h, x:x + w] == i).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = cv2.arcLength(cnts[0], True)
        if per == 0 or 4 * np.pi * a / per ** 2 < HOLE_MIN_CIRCULARITY:
            continue
        holes.append((i, a))
    if not holes:
        return np.zeros_like(mask), []
    # keep holes of a consistent size (rejects gaps between touching parts)
    med = np.median([a for _, a in holes])
    good = [i for i, a in holes if 0.35 * med <= a <= 2.5 * med]
    # a handful of "holes" in an image are gaps enclosed between touching
    # screws, not nuts: they are filled in and those blobs counted by size
    if len(good) < 5:
        filled = np.isin(lab, [i for i, _ in holes]).astype(np.uint8) * 255
        return filled, []
    hole_mask = np.isin(lab, good).astype(np.uint8) * 255
    return hole_mask, good


def single_object_area(areas):
    """Typical area of one loose object: iterate a trimmed median."""
    a = np.asarray(areas, float)
    if len(a) == 0:
        return 1.0
    med = np.median(a)
    for _ in range(5):
        sel = a[(a > 0.5 * med) & (a < 1.6 * med)]
        if len(sel) == 0:
            break
        med = np.median(sel)
    return med


def peak_seeds(dist, blob, k, min_sep, existing=()):
    """Pick up to k distance peaks inside a blob, at least min_sep away from
    each other and from already existing seeds (highest peaks first)."""
    d = np.where(blob, dist, 0)
    dil = cv2.dilate(d, np.ones((5, 5), np.uint8))
    ys, xs = np.where((d == dil) & (d > 1.5))
    order = np.argsort(-d[ys, xs])
    taken, chosen = list(existing), []
    for j in order:
        if len(chosen) >= k:
            break
        p = (xs[j], ys[j])
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep ** 2 for q in taken):
            taken.append(p)
            chosen.append(p)
    return chosen


def nearest_seed_labels(blob, seeds):
    """Assign every blob pixel to its nearest seed point (Voronoi cells)."""
    h, w = blob.shape
    seed_img = np.full((h, w), 255, np.uint8)
    for (px, py) in seeds:
        seed_img[py, px] = 0
    _, vor = cv2.distanceTransformWithLabels(seed_img, cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    # map voronoi label -> seed index
    lut = np.zeros(vor.max() + 1, np.int32)
    for idx, (px, py) in enumerate(seeds):
        lut[vor[py, px]] = idx + 1
    return np.where(blob, lut[vor], 0)


def fill_unlabelled(regions, blob):
    """Give foreground pixels the watershed left empty to the nearest region."""
    hole = blob & (regions == 0)
    if not hole.any() or not (regions > 0).any():
        return regions
    src = np.where(regions > 0, 0, 255).astype(np.uint8)
    _, near = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5,
                                              labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.where(regions > 0)
    lut = np.zeros(near.max() + 1, regions.dtype)
    lut[near[ys, xs]] = regions[ys, xs]
    out = regions.copy()
    out[hole] = lut[near[hole]]
    return out


def count_image(path):
    img, flat = load_and_flatten(path)
    mask = foreground(flat)
    hole_mask, nut_ids = nut_holes(mask)
    solid = cv2.bitwise_or(mask, hole_mask)  # nuts with holes filled
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)

    n, lab, st, _ = cv2.connectedComponentsWithStats(solid)
    hn, hlab, hst, hcen = cv2.connectedComponentsWithStats(hole_mask if nut_ids else np.zeros_like(hole_mask))

    # which hole centres fall into which blob
    holes_in = {i: [] for i in range(1, n)}
    for h in range(1, hn):
        cx, cy = np.round(hcen[h]).astype(int)
        holes_in[lab[cy, cx]].append((cx, cy)) if lab[cy, cx] else None

    loose = [st[i, cv2.CC_STAT_AREA] for i in range(1, n) if not holes_in[i]]
    unit = single_object_area(loose)
    single_nut = [st[i, cv2.CC_STAT_AREA] for i in range(1, n) if len(holes_in[i]) == 1]
    nut_unit = np.median(single_nut) if len(single_nut) >= 3 else None

    regions = np.zeros(solid.shape, np.int32)
    next_id = 1
    for i in range(1, n):
        x, y, w, h, area = st[i]
        sub = lab[y:y + h, x:x + w] == i
        d = dist[y:y + h, x:x + w]
        holes = [(cx - x, cy - y) for cx, cy in holes_in[i]]

        if holes:
            # nut blob: one cell per hole, plus extra seeds for covered holes
            ref = nut_unit or area / len(holes)
            extra = int(round((area - len(holes) * ref) / ref - 0.2))
            seeds = holes + (peak_seeds(d, sub, extra, 0.8 * np.sqrt(ref), holes)
                             if extra > 0 else [])
            local = nearest_seed_labels(sub, seeds)
        else:
            # loose parts: k seeds on distance peaks, watershed between them
            if area < 0.3 * unit:
                continue  # speckle / dust, far smaller than any real part
            k = max(1, int(round(area / unit - 0.15)))
            # overlapping parts can share one distance peak: if the size says
            # k objects but fewer seeds were found, relax the spacing and retry
            for sep in SEED_SEPARATION:
                seeds = peak_seeds(d, sub, k, sep * np.sqrt(unit))
                if len(seeds) >= k:
                    break
            if len(seeds) <= 1:
                local = sub.astype(np.int32)
            else:
                mk = np.zeros(sub.shape, np.int32)
                for s_i, (px, py) in enumerate(seeds):
                    cv2.circle(mk, (px, py), 2, s_i + 1, -1)
                mk[~sub] = len(seeds) + 1
                # flooding landscape: low in object centres (distance map) and
                # high on bright rims / highlights, which is where two
                # overlapping parts actually meet in the photo
                shape_term = 1.0 - cv2.normalize(d * sub, None, 0, 1, cv2.NORM_MINMAX)
                edge_term = flat[y:y + h, x:x + w].astype(np.float32) / 255.0
                land = (1 - EDGE_WEIGHT) * shape_term + EDGE_WEIGHT * edge_term
                ws_in = cv2.cvtColor(np.clip(land * 255, 0, 255).astype(np.uint8),
                                     cv2.COLOR_GRAY2BGR)
                local = cv2.watershed(ws_in, mk)
                local[(local < 0) | (local > len(seeds)) | ~sub] = 0
                local = fill_unlabelled(local, sub)

        local = np.where(sub & (mask[y:y + h, x:x + w] > 0), local, 0)
        nz = local > 0
        regions[y:y + h, x:x + w][nz] = local[nz] + next_id - 1
        next_id += int(local.max())

    ids = [r for r in np.unique(regions) if r > 0]
    return img, regions, ids, unit


def overlay(img, regions, ids, title):
    rng = np.random.default_rng(7)
    colors = rng.integers(40, 255, (max(ids, default=0) + 1, 3)).astype(np.uint8)
    out = img.copy()
    m = regions > 0
    out[m] = (0.45 * out[m] + 0.55 * colors[regions[m]]).astype(np.uint8)
    for r in ids:
        rm = (regions == r).astype(np.uint8)
        cnts, _ = cv2.findContours(rm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
    cv2.rectangle(out, (0, 0), (900, 70), (0, 0, 0), -1)
    cv2.putText(out, title, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(args.data, "*", "*.jpg"))):
        img, regions, ids, unit = count_image(path)
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.out, name + "_overlay.jpg"),
                    overlay(img, regions, ids, f"{name}: {len(ids)} objects"),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        np.save(os.path.join(args.out, name + "_regions.npy"), regions.astype(np.uint16))
        rows.append((name, len(ids)))
        print(f"{name:22s} {len(ids):5d}   (unit area {unit:.0f})")
    with open(os.path.join(args.out, "counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "count"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
