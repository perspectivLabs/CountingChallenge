"""
Non-AI Counting Challenge: classical computer vision (no learned model).

Run:   python count_nonai.py
       -> a file dialog opens, pick an image, a segmentation overlay is shown
          and a prediction report (Nuts / Screws) is printed.

Optional:  python count_nonai.py path/to/image.jpg   (skip the dialog)

Pipeline
  1. background correction (removes uneven lighting / vignetting)
  2. Otsu threshold -> foreground mask
  3. instance separation
       "holes": every nut has a see-through hole -> one watershed seed per hole
       "dist" : distance-transform watershed guided by a single-object area prior
  4. each instance is classed Nut (has a hole) or Screw (no hole)
"""
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# =========================================================
# 1. CONFIGURATION
# =========================================================
MODE = "auto"            # "auto" | "dist" | "holes"
POLARITY = "auto"        # "auto" | "dark" | "light" objects on the background
BG_FRAC = 0.30           # background kernel as fraction of image size; must be LARGER than the biggest object
THRESH_SCALE = 1.0       # <1 keeps fainter foreground, >1 is stricter (cuts shadows)
AUTO_HOLE_OBJECTS = 40   # in auto mode: >= this many see-through holes -> use "holes" mode
MIN_AGREEING_OBJECTS = 6 # clump splitting only if >= this many components have a similar size
FRAGMENT_FRAC = 0.05     # components smaller than this fraction of the largest are ignored when sizing one object
CLOSE_FRAC = 0.004       # closing kernel as fraction of image size (repairs shiny slits inside one object)
BAD_MASK_FRAC = 0.20     # warn if ONE foreground blob covers more than this fraction of the image
SAVE_DIR = Path("outputs")   # overlay image is saved here ("" / None to disable)
SHOW_PLOT = True

# =========================================================
# 2. CORE (mask, instance separation)
# =========================================================


# ----------------------------------------------------------------------------
# 1-2. foreground mask
# ----------------------------------------------------------------------------
def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def detect_polarity(gray):
    """Objects are the minority, so the background intensity ~ the median.
    Whichever side the intensity tail extends further is the object side."""
    med = np.median(gray)
    lo, hi = np.percentile(gray, [1, 99])
    return "dark" if (med - lo) >= (hi - med) else "light"


def foreground_mask(bgr, polarity="auto", bg_frac=0.05, thresh_scale=1.0):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if polarity == "auto":
        polarity = detect_polarity(gray)

    # estimate the background on a small copy (fast), then upsample
    s = 0.25 if max(gray.shape) > 800 else 1.0
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    k = _odd(max(9, bg_frac * min(small.shape)))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # dark objects -> closing removes them; light objects -> opening removes them
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE if polarity == "dark" else cv2.MORPH_OPEN, se)
    bg_small = cv2.GaussianBlur(bg_small, (_odd(k), _odd(k)), 0)
    bg = cv2.resize(bg_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)

    diff = (bg.astype(np.int16) - gray) if polarity == "dark" else (gray.astype(np.int16) - bg)
    sig = np.clip(diff, 0, 255).astype(np.uint8)
    sig = cv2.GaussianBlur(sig, (5, 5), 0)

    t, _ = cv2.threshold(sig, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(sig, t * thresh_scale, 255, cv2.THRESH_BINARY)

    k3 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)  # remove specks
    return mask > 0, polarity


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def estimate_single_area(areas):
    """Typical single-object area = the lower mode of the component areas."""
    areas = np.asarray(areas, dtype=float)
    if len(areas) == 0:
        return 1.0
    ref = np.percentile(areas, 25)
    for _ in range(3):
        sel = areas[(areas > 0.6 * ref) & (areas < 1.5 * ref)]
        if len(sel) == 0:
            break
        ref = np.median(sel)
    return float(ref)


def kmeans_split(comp_mask, k):
    ys, xs = np.nonzero(comp_mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, lbl, _ = cv2.kmeans(pts, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    out = np.zeros(comp_mask.shape, np.int32)
    out[ys, xs] = lbl.ravel() + 1
    return out


# ----------------------------------------------------------------------------
# 3a. distance-transform mode (screws, bolts)
# ----------------------------------------------------------------------------
def split_dist(mask, min_area_frac=0.35, split_thresh=1.6):
    """Returns an int32 label image, one label per instance."""
    kc = _odd(max(5, round(CLOSE_FRAC * min(mask.shape))))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))) > 0
    mask = ndi.binary_fill_holes(mask)

    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    floor = max(30, 2e-5 * mask.size)
    big = areas[areas >= floor]
    if len(big) == 0:
        return np.zeros(mask.shape, np.int32), 0.0
    # specks and thread fragments must not define "one object": only components that are a
    # meaningful fraction of the largest one take part in estimating the single-object size
    big = big[big >= FRAGMENT_FRAC * big.max()]
    single = estimate_single_area(big)
    # The area prior is only meaningful when many components look alike (a scatter of identical
    # parts). With a few big, different-sized objects (e.g. a catalogue photo of 3-7 wood screws)
    # it is not, and splitting would just slice objects into pieces -> disable splitting.
    n_agree = int(((big > 0.6 * single) & (big < 1.5 * single)).sum())
    allow_split = n_agree >= MIN_AGREEING_OBJECTS

    # typical inscribed radius of a single object (for min peak spacing)
    dist_all = ndi.distance_transform_edt(mask)
    radii = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if 0.6 * single < a < 1.5 * single:
            radii.append(dist_all[cc == i].max())
    r_typ = float(np.median(radii)) if radii else max(3.0, np.sqrt(single / np.pi) * 0.6)
    min_dist = max(3, int(0.6 * r_typ))

    out = np.zeros(mask.shape, np.int32)
    nxt = 1
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area_frac * single:
            continue  # noise
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        sl = (slice(y, y + h), slice(x, x + w))
        comp = cc[sl] == i
        k = int(round(a / single))
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = max(1.0, cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))))
        solidity = a / hull_area
        # big AND non-convex (a clump) -> split; a big convex blob is just one big object
        if (not allow_split) or a < split_thresh * single or k <= 1 or solidity > 0.93:
            out[sl][comp] = nxt
            nxt += 1
            continue

        d = ndi.gaussian_filter(ndi.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1], 1.0)
        peaks = peak_local_max(d, min_distance=min_dist, num_peaks=k,
                               labels=comp.astype(np.int32), exclude_border=False)
        if len(peaks) >= 2:
            markers = np.zeros(comp.shape, np.int32)
            for j, (py, px) in enumerate(peaks, 1):
                markers[py, px] = j
            lab = watershed(-d, markers, mask=comp)
            parts = len(peaks)
        else:  # fallback: area-driven k-means
            lab = kmeans_split(comp, k)
            parts = k
        for j in range(1, parts + 1):
            m = lab == j
            if m.sum() >= min_area_frac * single:
                out[sl][m] = nxt
                nxt += 1
    return out, single


# ----------------------------------------------------------------------------
# 3b. hole-seeded mode (nuts / washers / anything with a see-through hole)
# ----------------------------------------------------------------------------
def split_holes(mask, min_area_frac=0.35):
    raw = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    filled = ndi.binary_fill_holes(raw)
    holes = filled & ~raw

    hl, hn = ndi.label(holes)
    if hn == 0:
        return split_dist(mask)  # no holes found, fall back
    h_areas = ndi.sum(holes, hl, index=np.arange(1, hn + 1))
    h_floor = max(6, 1e-6 * mask.size)
    good = h_areas >= h_floor
    if not good.any():
        return split_dist(mask)
    h_ref = np.median(h_areas[good])
    keep_mask = good & (h_areas > 0.3 * h_ref) & (h_areas < 3.0 * h_ref)
    # a nut's hole is small and round; the gap between 3-4 touching nuts is a star -> reject by roundness
    from skimage.measure import regionprops
    for rp in regionprops(hl):
        idx = rp.label - 1
        if not keep_mask[idx]:
            continue
        circ = 4 * np.pi * rp.area / max(1.0, rp.perimeter) ** 2
        if rp.area >= 12 and (circ < 0.55 or rp.solidity < 0.85):
            keep_mask[idx] = False
    keep = np.where(keep_mask)[0] + 1

    markers = np.zeros(mask.shape, np.int32)
    for j, lab_id in enumerate(keep, 1):
        markers[hl == lab_id] = j
    n_mark = len(keep)

    # --- reference nut size from ISOLATED nuts (blobs that contain exactly one hole seed)
    cc, cn = ndi.label(filled)
    seeds_per_cc = np.bincount(cc[markers > 0], minlength=cn + 1)   # pixels, not seeds -> count seeds below
    seed_cc = {}
    for j in range(1, n_mark + 1):
        ys, xs = np.nonzero(markers == j)
        seed_cc.setdefault(int(cc[ys[0], xs[0]]), []).append(j)
    areas_single = [int((cc == c).sum()) for c, js in seed_cc.items() if len(js) == 1]
    if len(areas_single) >= 3:
        ref_area = float(np.median(areas_single))
    else:
        ref_area = float(np.median([(cc == c).sum() / max(1, len(js)) for c, js in seed_cc.items()]))
    r_eq = np.sqrt(ref_area / np.pi)
    R = 1.25 * r_eq   # a nut never extends further than this from its hole

    # --- every foreground pixel goes to its nearest hole, if within one nut radius
    dist_seed, (iy, ix) = ndi.distance_transform_edt(markers == 0, return_indices=True)
    valid = raw | (markers > 0)          # real object pixels + the seed holes (not the gaps between nuts)
    lab = np.where(valid & (dist_seed <= R), markers[iy, ix], 0).astype(np.int32)

    # --- leftover foreground = nuts whose hole is hidden by overlap (or odd objects)
    resid = valid & (lab == 0)
    resid = cv2.morphologyEx(resid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    rl, rn = ndi.label(resid)
    nxt = int(lab.max()) + 1
    for c in range(1, rn + 1):
        m = rl == c
        a = int(m.sum())
        if a < 0.6 * ref_area:
            continue
        cn_, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = max(1.0, cv2.contourArea(cv2.convexHull(max(cn_, key=cv2.contourArea))))
        if a < 1.5 * ref_area and a / hull < 0.9:
            continue   # concave sliver left between nuts, not a nut
        k = max(1, int(round(a / ref_area)))
        if k == 1:
            lab[m] = nxt
            nxt += 1
            continue
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = kmeans_split(m[y0:y1, x0:x1], k)
        for j in range(1, k + 1):
            part = sub == j
            if part.sum() >= 0.35 * ref_area:
                lab[y0:y1, x0:x1][part] = nxt
                nxt += 1
    return lab, ref_area


# ----------------------------------------------------------------------------
# 4. overlay
# ----------------------------------------------------------------------------
# =========================================================
# 3. NUT / SCREW CLASSIFICATION, OVERLAY
# =========================================================
def find_holes(mask):
    raw = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    holes = ndi.binary_fill_holes(raw) & ~raw
    hl, hn = ndi.label(holes)
    if hn == 0:
        return hl, np.array([])
    areas = ndi.sum(holes, hl, index=np.arange(1, hn + 1))
    return hl, areas


def count_good_holes(areas, mask_size):
    if len(areas) == 0:
        return 0
    good = areas[areas >= max(6, 1e-6 * mask_size)]
    if len(good) == 0:
        return 0
    ref = np.median(good)
    return int(((good > 0.3 * ref) & (good < 3.0 * ref)).sum())


def classify_instances(labels, hl, h_areas, mode):
    """Nut  = compact object (aspect < 1.5) whose see-through hole is a real fraction
              of its area (a nut's hole is ~15-25 percent; shiny-head slivers are tiny).
    Screw = everything else (elongated, or no sizeable hole).
    In the dense-pile mode, hole-less compact pieces are still nuts (hole hidden by overlap)."""
    kinds = {}
    holes_any = hl > 0
    for i in [i for i in np.unique(labels) if i != 0]:
        m = labels == i
        area = int(m.sum())
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        aspect = max(w, h) / max(1.0, min(w, h))
        hole_frac = float((holes_any & m).sum()) / max(1, area)
        compact = aspect < 1.5
        big_hole = hole_frac >= 0.15 and aspect < 2.5   # large see-through hole: a nut, even if the cut is oddly shaped
        is_nut = big_hole or (compact and (hole_frac >= 0.08 or mode == "holes"))
        kinds[i] = "Nuts" if is_nut else "Screws"
    return kinds


def make_overlay(bgr, labels, kinds):
    out = bgr.copy()
    rng = np.random.default_rng(3)
    for i, kind in kinds.items():
        m = labels == i
        color = rng.integers(60, 255, 3)
        out[m] = (0.45 * out[m] + 0.55 * color).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 0, 255), 1)
    return out


def run(bgr):
    mask, polarity = foreground_mask(bgr, POLARITY, BG_FRAC, THRESH_SCALE)
    hl, h_areas = find_holes(mask)
    mode = MODE
    if mode == "auto":
        mode = "holes" if count_good_holes(h_areas, mask.size) >= AUTO_HOLE_OBJECTS else "dist"
    run.warnings = []
    n_cc, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_cc > 1 and st[1:, cv2.CC_STAT_AREA].max() > BAD_MASK_FRAC * mask.size:
        run.warnings.append(
            "One foreground blob covers %.0f%% of the image. The background is probably textured, "
            "shadowed or unevenly lit, so a global threshold cannot separate the objects and this "
            "count is unreliable. Use the AI method for this image." % (100 * st[1:, cv2.CC_STAT_AREA].max() / mask.size))
    labels, _ = split_holes(mask) if mode == "holes" else split_dist(mask)
    kinds = classify_instances(labels, hl, h_areas, mode)
    return labels, kinds, mode, polarity


# =========================================================
# 4. SELECT IMAGE FROM LOCAL COMPUTER
# =========================================================
def get_local_image_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")],
        )
        root.update()
        root.destroy()
        return path
    except Exception:
        return input("Path to image: ").strip().strip('"')


def main():
    image_path = get_local_image_path()
    if not image_path:
        print("No image selected.")
        raise SystemExit
    filename = os.path.basename(image_path)
    print("\nSelected image:", filename)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")

    t0 = time.perf_counter()
    labels, kinds, mode, polarity = run(bgr)
    elapsed = time.perf_counter() - t0

    counts = Counter(kinds.values())
    nuts, screws = counts["Nuts"], counts["Screws"]
    overlay = make_overlay(bgr, labels, kinds)

    if SAVE_DIR:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SAVE_DIR / f"{Path(filename).stem}_overlay.png"
        cv2.imwrite(str(out_path), overlay)

    if SHOW_PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 10))
        plt.imshow(overlay[:, :, ::-1])
        plt.title(f"Non-AI Segmentation: {filename}   |   Total: {nuts + screws}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 45)
    print("           PREDICTION REPORT")
    print("=" * 45)
    print("Image:", filename)
    print("Method: classical CV (no learned model)")
    print("Separation mode:", mode, f"(objects are {polarity} on background)")
    print("-" * 45)
    print("Nuts detected:", nuts)
    print("Screws detected:", screws)
    print("Total objects:", nuts + screws)
    print("-" * 45)
    print(f"Processing time: {elapsed * 1000:.2f} ms")
    for w in getattr(run, "warnings", []):
        print("WARNING:", w)
    if SAVE_DIR:
        print("Overlay saved to:", out_path)
    print("=" * 45)


if __name__ == "__main__":
    main()
