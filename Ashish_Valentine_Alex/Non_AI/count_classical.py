"""Classical (non-AI) screw counter: background normalisation + threshold + watershed."""
import argparse, csv, glob, os
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import h_maxima

WORK_LONG_SIDE = 1600


def segment(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = WORK_LONG_SIDE / max(h, w)
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # background estimate: big median blur removes small dark objects, keeps vignetting/gradients
    k = int(max(small.shape[:2]) * 0.08) | 1
    bg = cv2.medianBlur(gray, k)
    diff = cv2.subtract(bg, gray)  # objects are darker than background
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # guard against Otsu picking noise on nearly empty diffs
    t = max(_, 25)
    _, mask = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8)
    return small, mask


def robust_ref(areas, start_pct=35):
    """Typical single-object area. Starts from a low percentile (merged pairs inflate the median),
    then re-centres on the blobs that sit near that size."""
    areas = areas[areas > 60]
    if len(areas) == 0:
        return 1.0
    ref = np.percentile(areas, start_pct)
    for _ in range(3):
        sel = areas[(areas > 0.6 * ref) & (areas < 1.4 * ref)]
        if len(sel):
            ref = np.median(sel)
    return float(ref)


def split_and_count(mask, h=1.4, frag=0.35, min_blob=0.3, big=1.7):
    """Count objects in a foreground mask and return an instance-label map.

    1. Watershed seeded by h-maxima of the smoothed distance transform: separates touching objects whose
       thickest parts are clearly distinct (e.g. an end-on disc next to a lying screw), but a screw's thin
       shank does not create a peak of its own.
    2. Fragments smaller than `frag` x a typical object are merged into the neighbour they share the longest
       border with.
    3. The single-object area is re-measured from the resulting segments. Any segment still larger than
       `big` x that (tight clumps of overlapping screws) is split into round(area / single area) parts with
       k-means on its pixel coordinates. Overlap only hides area, so this can undercount but not overcount.
    """
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, np.int32), 0
    ref0 = robust_ref(st[1:, cv2.CC_STAT_AREA])
    dt = cv2.GaussianBlur(cv2.distanceTransform(mask, cv2.DIST_L2, 5), (0, 0), 1.0)
    markers, _ = ndi.label(h_maxima(dt, h) & (mask > 0))
    ws = watershed(-dt, markers, mask=mask > 0)
    for i in [i for i in np.unique(ws) if i > 0 and (ws == i).sum() < frag * ref0]:
        m = ws == i
        ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        nb = ws[ring & ~m]
        nb = nb[(nb > 0) & (nb != i)]
        if len(nb):
            ws[m] = np.bincount(nb).argmax()
    ids = [i for i in np.unique(ws) if i > 0]
    areas = np.array([(ws == i).sum() for i in ids])
    ref = robust_ref(areas)
    out = np.zeros(mask.shape, np.int32)
    k = 0
    for i, a in zip(ids, areas):
        if a < min_blob * ref:
            continue
        seg = ws == i
        n_est = int(round(a / ref))
        if a < big * ref or n_est < 2:
            k += 1
            out[seg] = k
            continue
        ys, xs = np.nonzero(seg)
        pts = np.float32(np.c_[xs, ys])
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        cv2.setRNGSeed(0)
        _, lb, _ = cv2.kmeans(pts, n_est, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        for j in range(n_est):
            k += 1
            sel = lb.ravel() == j
            out[ys[sel], xs[sel]] = k
    return out, k


def overlay(small, labels, count):
    rng = np.random.RandomState(1)
    colors = rng.randint(60, 255, (labels.max() + 1, 3))
    colors[0] = 0
    col = colors[labels].astype(np.uint8)
    blend = np.where(labels[..., None] > 0, (0.5 * small + 0.5 * col).astype(np.uint8), small)
    cv2.putText(blend, f"count={count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    return blend


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.input, "**", "*.jpg"), recursive=True)):
        img = cv2.imread(f)
        small, mask = segment(img)
        labels, n = split_and_count(mask)
        name = os.path.relpath(f, a.input).replace(os.sep, "_")
        cv2.imwrite(os.path.join(a.output, name), overlay(small, labels, n))
        rows.append((os.path.relpath(f, a.input), n))
        print(rows[-1])
    with open(os.path.join(a.output, "counts.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows([("image", "count")] + rows)
