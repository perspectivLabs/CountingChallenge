"""Non-AI object counting with classical OpenCV.

Pipeline:
  1. Downscale so the long side is ~1600 px (speed + consistent kernel sizes).
  2. Flatten illumination: divide grayscale by a heavily blurred background estimate.
  3. Otsu threshold (objects are darker than the background), then morphological clean-up
     and hole filling (washers have holes).
  4. Connected components -> blobs. Small specks are discarded.
  5. Nuts: every round, consistently sized hole inside a blob is one nut (gaps enclosed
     between touching parts are irregular and rejected by a circularity test). Touching
     nuts are split into instance masks by nearest-hole-centre (Voronoi) assignment.
  6. Screws/bolts (blobs without holes): the single-object area is the smallest dominant
     mode of the blob-area histogram; a cluster counts as round(area / single_area) and
     is split into masks with distance-transform watershed.
     Round discs (bigger than a screw) are modelled separately: isolated round blobs give
     the disc area and radius; inside a cluster every distance-transform core thicker than
     ~0.8 disc radius is a disc. The remaining area is divided by the screw area, with a
     6% crowding allowance for clusters (shadows/edges between touching objects add area;
     measured 1.05-1.07 per object on clusters of 3+).
  7. Draw a coloured mask per detected instance and write the count on the image.
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np


def load_resized(path, long_side=1600):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def foreground_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = int(max(img.shape[:2]) / 8) | 1
    # background = closing (removes dark objects) then big blur
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    bg = cv2.GaussianBlur(bg, (k, k), 0)
    norm = np.clip(gray / (bg + 1e-6) * 255, 0, 255).astype(np.uint8)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)
    _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # fill holes (washers / shiny highlights)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(th)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    # valid field of view: excludes the black lens-vignette corners in some shots
    valid = (bg > 0.55 * np.median(bg)).astype(np.uint8)
    valid = cv2.erode(valid, np.ones((25, 25), np.uint8))
    return filled, norm, valid


def valid_components(labels, stats, valid):
    """Drop blobs that are mostly outside the valid (non-vignette) field of view."""
    n = stats.shape[0]
    keep = np.ones(n, bool)
    inside = np.bincount(labels.ravel(), weights=valid.ravel().astype(np.float64), minlength=n)
    frac = inside / np.maximum(stats[:, cv2.CC_STAT_AREA], 1)
    keep[frac < 0.7] = False
    keep[0] = False
    return keep


def single_object_area(areas):
    """Area of one object = the smallest well-populated mode of the log-area histogram.

    The median fails when most blobs are touching pairs (e.g. sparse scenes where screws
    lie in pairs). Singles form the left-most significant peak.
    """
    la = np.log(areas[areas > 20])
    if len(la) < 3:
        return float(np.median(areas))
    bins = np.linspace(la.min(), la.max() + 1e-6, 40)
    hist, edges = np.histogram(la, bins)
    hist = np.convolve(hist, [1, 2, 3, 2, 1], "same").astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    peaks = [i for i in range(len(hist)) if hist[i] > 0 and hist[i] >= hist[max(i - 1, 0)] and hist[i] >= hist[min(i + 1, len(hist) - 1)]]
    peaks = [i for i in peaks if hist[i] >= 0.35 * hist.max()]
    c = np.exp(centers[peaks[0]])
    near = areas[(areas > 0.7 * c) & (areas < 1.4 * c)]
    return float(np.median(near)) if len(near) else float(c)


def disc_model(labels, stats, keep, single):
    """(area, radius) of round discs from isolated round blobs; None if the scene has none."""
    rs, areas = [], []
    for i in range(1, stats.shape[0]):
        a = stats[i, cv2.CC_STAT_AREA]
        if not keep[i] or not 0.8 * single < a < 2.5 * single:
            continue
        x, y, w, h = stats[i, :4]
        b = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        c = max(cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        p = cv2.arcLength(c, True)
        if p > 0 and 4 * np.pi * cv2.contourArea(c) / (p * p) > 0.8:
            rs.append(cv2.distanceTransform(np.pad(b, 1), cv2.DIST_L2, 5).max())
            areas.append(a)
    if len(rs) < 3:
        return None
    return float(np.median(areas)), float(np.median(rs))


def count_discs(blob, r):
    """Discs inside a blob: cores where the distance to the edge is >= 0.8 disc radius
    (screw shanks/heads are much thinner). Touching discs give an elongated core."""
    dt = cv2.distanceTransform(np.pad(blob, 1), cv2.DIST_L2, 5)
    nc, lab = cv2.connectedComponents((dt >= 0.8 * r).astype(np.uint8), 8)
    k = 0
    for j in range(1, nc):
        ys, xs = np.nonzero(lab == j)
        k += 1 + int(max(np.ptp(xs), np.ptp(ys)) // (1.6 * r))
    return k


def split_cluster(blob_mask, n_expected):
    """Watershed split of one blob; returns label image (1..k)."""
    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    best = None
    for frac in (0.6, 0.5, 0.4, 0.3, 0.2):
        _, peaks = cv2.threshold(dist, frac * dist.max(), 255, cv2.THRESH_BINARY)
        n, markers = cv2.connectedComponents(peaks.astype(np.uint8))
        best = (n - 1, markers)
        if n - 1 >= n_expected:
            break
    k, markers = best
    if k <= 1:
        return np.where(blob_mask > 0, 1, 0).astype(np.int32), 1
    markers = markers.astype(np.int32)
    markers[(blob_mask > 0) & (markers == 0)] = 0
    markers[blob_mask == 0] = -1  # placeholder
    rgb = cv2.cvtColor(blob_mask, cv2.COLOR_GRAY2BGR)
    m = markers.copy()
    m[m == -1] = k + 1  # background marker
    cv2.watershed(rgb, m)
    m[(m == k + 1) | (m == -1) | (blob_mask == 0)] = 0
    return m, k


def find_round_holes(th):
    """Inner contours (holes) of the foreground that are round and of consistent size.

    Hex nuts have one round through-hole each; enclosed gaps between touching objects
    are irregular and are rejected by the circularity test.
    Returns (list of (cx, cy, contour), typical hole area).
    """
    cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cand = []
    for i, c in enumerate(cnts):
        if hier is None or hier[0][i][3] < 0:
            continue
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if a < 30 or p == 0:
            continue
        if 4 * np.pi * a / (p * p) > 0.75:
            cand.append((a, c))
    if len(cand) < 5:  # not a nut scene
        return [], 0.0
    med = float(np.median([a for a, _ in cand]))
    holes = []
    for a, c in cand:
        if 0.4 * med < a < 2.0 * med:
            m = cv2.moments(c)
            holes.append((m["m10"] / m["m00"], m["m01"] / m["m00"], c))
    return holes, med


def count_image(path, min_area_frac=0.25, crowd=1.06):
    img = load_resized(path)
    mask, norm, valid = foreground_mask(img)
    _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return img, 0, []
    keep = valid_components(labels, stats, valid)

    # --- nuts: one round hole == one nut -------------------------------------------
    holes, _ = find_round_holes(th)
    holes_in = {}
    for hx, hy, c in holes:
        lab = labels[int(round(hy)), int(round(hx))]
        if lab > 0 and keep[lab]:
            holes_in.setdefault(lab, []).append(c)
    one_hole = [stats[i, cv2.CC_STAT_AREA] for i, hs in holes_in.items() if len(hs) == 1]
    nut_area = float(np.median(one_hole)) if one_hole else 0.0

    # --- other objects (screws/bolts): area of a single object ------------------------
    plain = np.array([stats[i, cv2.CC_STAT_AREA] for i in range(1, n) if keep[i] and i not in holes_in])
    single = single_object_area(plain) if len(plain) else max(nut_area, 1.0)
    discs = disc_model(labels, stats, keep, single) if not holes_in else None

    overlay = img.copy()
    total = 0
    rng = np.random.default_rng(0)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if not keep[i] or a < min_area_frac * min(single, nut_area or single):
            continue
        x, y, w, h = stats[i, :4]
        blob = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        hs = holes_in.get(i, [])
        if hs:
            k = len(hs)
            rest = a - k * nut_area
            if len(plain) and rest > 0.6 * single:
                k += int(round(rest / single))  # e.g. a screw touching a nut
            # nuts are equal-sized convex parts: nearest-hole-centre (Voronoi) split
            ctr = np.array([cv2.moments(c)["m10"] / cv2.moments(c)["m00"] - x for c in hs]), \
                np.array([cv2.moments(c)["m01"] / cv2.moments(c)["m00"] - y for c in hs])
            yy, xx = np.nonzero(blob)
            d = (xx[:, None] - ctr[0][None]) ** 2 + (yy[:, None] - ctr[1][None]) ** 2
            parts = np.zeros(blob.shape, np.int32)
            parts[yy, xx] = np.argmin(d, 1) + 1
        else:
            nd = count_discs((blob > 0).astype(np.uint8), discs[1]) if discs else 0
            k0 = (a - nd * discs[0]) / single if discs else a / single
            if nd + k0 > 2.5:  # cluster: allow for shadow/edge area between objects
                k0 /= crowd
            k = max(1, nd + (int(round(k0)) if (k0 > 0.5 or nd == 0) else 0))
            if k == 1:
                parts = (blob > 0).astype(np.int32)
            else:
                parts, _ = split_cluster(np.pad(blob, 2), k)
                parts = parts[2:-2, 2:-2]
        total += k
        roi = overlay[y:y + h, x:x + w]
        for j in [j for j in np.unique(parts) if j > 0]:
            c = rng.integers(60, 255, 3)
            sel = parts == j
            roi[sel] = (0.45 * roi[sel] + 0.55 * c).astype(np.uint8)
            cs, _ = cv2.findContours((sel * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(int(x), int(y)))
            cv2.drawContours(overlay, cs, -1, (30, 30, 30), 1)
        if k > 1 and not hs:
            cv2.putText(overlay, str(k), (int(x), int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.rectangle(overlay, (0, 0), (360, 50), (255, 255, 255), -1)
    cv2.putText(overlay, f"Count: {total}", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    return overlay, total, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "outputs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(args.data, "**", "*.jpg"), recursive=True)):
        overlay, total, _ = count_image(p)
        name = os.path.basename(p)
        cv2.imwrite(os.path.join(args.out, name), overlay)
        rows.append((os.path.relpath(p, args.data), total))
        print(f"{name}: {total}")
    with open(os.path.join(args.out, "counts.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("image", "count"), *rows])


if __name__ == "__main__":
    main()
