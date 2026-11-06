"""
Non-AI object counting + instance masks for the perspectivLabs CountingChallenge.

Pipeline (classical computer vision only: OpenCV + NumPy + SciPy):
  1. Flat-field correction  - estimate the background with a large morphological
     closing and divide it out, removing uneven lighting and lens vignetting.
  2. Otsu thresholding      - objects are darker than the background.
  3. Vignette removal       - drop dark blobs that run along the image border.
  4. Nut counting           - hex nuts touch each other constantly, but every nut has
     a round through-hole. If an image contains many circular holes of a consistent
     size, each hole seeds one nut; nut pixels are assigned to the nearest hole.
  5. Area-based counting    - for everything else (screws, bolts), merged blobs are
     counted as round(blob_area / single_object_area). The single-object area is
     estimated per image with an iterated median, so it adapts to scale. Round,
     solid blobs (screws standing upright) get their own reference area, and inside
     merged clusters they are found as thick distance-transform cores and counted
     separately. Clusters larger than 4 objects round with a -0.2 bias, since overlap hides area.
  6. Instance masks         - a blob counted as n objects is split into n masks by
     k-means on (x, y, local orientation), so crossing screws split along their axes.

Usage:
  python count_objects.py --input <images_dir> --output results [--gt ../ground_truth.csv] [--save-masks]
"""
import argparse, csv, glob, os, time
import cv2
import numpy as np
from scipy import ndimage as ndi

def load(path, long_side=2000):
    im = cv2.imread(path)
    s = min(1.0, long_side / max(im.shape[:2]))
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

def foreground(im):
    """Flat-field the background, then Otsu. Objects are darker than background."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = int(max(g.shape) * 0.06) | 1
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4)
    n8 = (255 * np.clip(g / (bg + 1e-6), 0, 1)).astype(np.uint8)
    n8 = cv2.GaussianBlur(n8, (3, 3), 0)
    _, b = cv2.threshold(n8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return b > 0, n8

def find_holes(fg):
    """Background regions fully enclosed by foreground (e.g. nut holes)."""
    holes = ndi.binary_fill_holes(fg) & ~fg
    nl, lab, st, cen = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    out = []
    for i in range(1, nl):
        a = st[i, 4]
        if a < 12: continue
        m = (lab[st[i,1]:st[i,1]+st[i,3], st[i,0]:st[i,0]+st[i,2]] == i).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        p = cv2.arcLength(cs[0], True)
        circ = 4 * np.pi * a / (p * p + 1e-6)
        out.append((i, a, circ, cen[i]))
    return lab, out

def robust_unit(areas):
    """Typical single-object area: iterate median over a window around itself."""
    a = np.asarray(areas, float)
    u = np.median(a)
    for _ in range(10):
        sel = a[(a > 0.6 * u) & (a < 1.45 * u)]
        if len(sel) < 3: break
        nu = np.median(sel)
        if abs(nu - u) < 1: break
        u = nu
    return u

def nut_seeds(fg):
    """Return hole label image + list of hole ids that look like nut holes, or [] if not a nut image."""
    lab, holes = find_holes(fg)
    circ = [h for h in holes if h[2] > 0.70]
    if len(circ) < 30: return lab, []
    med = np.median([h[1] for h in circ])
    a = np.array([h[1] for h in circ])
    spread = (np.percentile(a, 90) - np.percentile(a, 10)) / med
    if med < 60 or spread > 0.6: return lab, []
    keep = [h for h in holes if 0.6 * med <= h[1] <= 1.6 * med and h[2] > 0.70]  # gaps between 3 touching nuts are less round
    return lab, keep

def local_orientation(mask, sigma):
    """Per-pixel dominant orientation (doubled-angle unit vector) from the mask's structure tensor."""
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma / 3)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0); gy = cv2.Sobel(m, cv2.CV_32F, 0, 1)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma); jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    c, s = jxx - jyy, 2 * jxy
    norm = np.sqrt(c * c + s * s) + 1e-6
    return c / norm, s / norm

def split_component(mask, n, scale=10.0, seed=0):
    """Split one blob into n instances: k-means on (x, y, orientation).
    Orientation lets crossing / side-by-side screws separate along their own axes."""
    ys, xs = np.nonzero(mask)
    if n <= 1 or len(xs) < n: return [mask]
    pad = np.pad(mask, 10)
    c, s = local_orientation(pad, sigma=max(2.0, scale / 2))
    c, s = c[10:-10, 10:-10], s[10:-10, 10:-10]
    w = scale * 0.8
    pts = np.column_stack([xs, ys, w * c[ys, xs], w * s[ys, xs]]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.3)
    cv2.setRNGSeed(seed)
    _, lbl, _ = cv2.kmeans(pts, n, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    out = []
    for k in range(n):
        m = np.zeros_like(mask); sel = lbl.ravel() == k
        m[ys[sel], xs[sel]] = True
        # keep largest connected piece; give stray pixels back to nearest instance later
        nl, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        if nl > 2:
            keep = 1 + np.argmax(st[1:, 4]); m = lab == keep
        out.append(m)
    # reassign orphan pixels to the nearest instance so the union still covers the blob
    lab = np.zeros(mask.shape, np.int32)
    for k, m in enumerate(out, 1): lab[m] = k
    orphan = mask & (lab == 0)
    if orphan.any():
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab[orphan] = lab[iy[orphan], ix[orphan]]
        out = [lab == k for k in range(1, n + 1)]
    return out

BIAS_MIN_UNITS = 4.0   # clusters bigger than this many objects get the overlap bias

def drop_vignette(mask):
    """Remove dark lens-vignette regions: blobs that run a long way along the image border."""
    nl, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if nl <= 1: return mask
    unit = robust_unit(st[1:, 4][st[1:, 4] > 30]) if (st[1:, 4] > 30).sum() else 100
    edge = np.zeros_like(mask); edge[[0, -1], :] = True; edge[:, [0, -1]] = True
    contact = np.bincount(lab[edge], minlength=nl)
    out = mask.copy()
    for i in range(1, nl):
        if contact[i] > 0.8 * np.sqrt(unit):
            out[lab == i] = False
    return out

def count_image(path, long_side=2000, round_bias=-0.2):
    im = load(path, long_side)
    fg, _ = foreground(im)
    H, W = fg.shape
    instances, kinds = [], []

    # 1) Nuts: one instance per hole; claim foreground pixels within a ring around each hole
    hole_lab, seeds = nut_seeds(fg)
    claimed = np.zeros_like(fg)
    if seeds:
        r_hole = np.sqrt(np.median([h[1] for h in seeds]) / np.pi)
        R = 2.6 * r_hole                       # outer nut radius ~ 2.3-2.6x hole radius
        markers = np.zeros((H, W), np.int32)
        for k, (hid, *_ ) in enumerate(seeds, 1):
            markers[hole_lab == hid] = k
        # distance from every pixel to nearest seed + which seed
        dist, (iy, ix) = ndi.distance_transform_edt(markers == 0, return_indices=True)
        owner = markers[iy, ix]
        holes_all = ndi.binary_fill_holes(fg)
        region = holes_all & (dist <= R - r_hole)
        for k in range(1, len(seeds) + 1):
            m = region & (owner == k)
            instances.append(m); kinds.append('nut')
        claimed = region

    # 2) Everything else: area-based counting on remaining blobs
    rest = fg & ~cv2.dilate(claimed.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    rest = cv2.morphologyEx(rest.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    rest = ndi.binary_fill_holes(rest)
    rest = drop_vignette(rest)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), 8)
    areas = st[1:, 4]
    min_area = 0
    big = areas[areas > 30]
    if len(big) >= 3 or (len(big) and not seeds):
        unit = robust_unit(big) if not seeds else np.median([np.sum(m) for m in instances])
        min_area = 0.25 * unit
        # shape descriptors: round + solid blobs are upright screws / bolt heads seen from above
        disc = {}
        for i in range(1, nl):
            if st[i, 4] < min_area: continue
            x, y, w, h = st[i, :4]
            m = (lab[y:y+h, x:x+w] == i).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=cv2.contourArea)
            per = cv2.arcLength(c, True); hull = cv2.contourArea(cv2.convexHull(c)) + 1e-6
            circ = 4 * np.pi * st[i, 4] / (per * per + 1e-6)
            disc[i] = circ > 0.75 and st[i, 4] / hull > 0.93
        disc_areas = [st[i, 4] for i, d in disc.items() if d]
        disc_unit = np.median(disc_areas) if len(disc_areas) >= 3 else None
        for i in range(1, nl):
            a = st[i, 4]
            if a < min_area: continue
            x, y, w, h = st[i, :4]
            if seeds and a < 0.5 * unit: continue      # slivers of nuts left after claiming
            sub = lab[y:y+h, x:x+w] == i
            if disc.get(i) and disc_unit:
                n = max(1, int(np.floor(a / disc_unit + 0.5)))
            elif disc_unit and a > 1.5 * unit:
                # upright screws inside a cluster: their thick round cores survive a
                # distance-transform threshold that thin lying screws never reach
                r_d = np.sqrt(disc_unit / np.pi)
                dt = cv2.distanceTransform(np.pad(sub, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
                nc, _, cst, _ = cv2.connectedComponentsWithStats((dt >= 0.7 * r_d).astype(np.uint8), 8)
                k = int(np.sum(cst[1:, 4] >= 3)) if nc > 1 else 0
                rest_a = max(0.0, a - k * disc_unit)
                n = max(1, k + int(np.floor(rest_a / unit + 0.5 + (round_bias if rest_a > BIAS_MIN_UNITS * unit else 0))))
            else:
                # overlap hides area only in large clusters (> BIAS_MIN_UNITS objects): bias those, not pairs or small groups
                b = round_bias if a > BIAS_MIN_UNITS * unit else 0.0
                n = max(1, int(np.floor(a / unit + 0.5 + b)))
            for m in split_component(sub, n, scale=np.sqrt(unit)):
                full = np.zeros_like(fg); full[y:y+h, x:x+w] = m
                instances.append(full); kinds.append('obj')
    return im, instances, kinds

def overlay(im, instances, title=None):
    out = im.copy()
    rng = np.random.default_rng(7)
    col = rng.integers(40, 255, (len(instances), 3))
    for m, c in zip(instances, col):
        out[m] = (0.45 * out[m] + 0.55 * c).astype(np.uint8)
    fs = max(im.shape[:2]) / 2000
    for k, m in enumerate(instances, 1):
        ys, xs = np.nonzero(m)
        if len(xs) == 0: continue
        cv2.putText(out, str(k), (int(xs.mean()) - 6, int(ys.mean()) + 4), 0, 0.33 * fs, (255, 255, 255), 1, cv2.LINE_AA)
    txt = f"Count: {len(instances)}" if title is None else title
    cv2.rectangle(out, (0, 0), (int(330 * fs), int(55 * fs)), (0, 0, 0), -1)
    cv2.putText(out, txt, (int(10 * fs), int(40 * fs)), 0, 1.2 * fs, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def load_gt(path):
    gt = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("gt_count", "").strip():
                    gt[r["image"].strip()] = int(r["gt_count"])
    return gt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder with images (searched recursively)")
    ap.add_argument("--output", default="results")
    ap.add_argument("--gt", default=None, help="CSV with columns image,gt_count")
    ap.add_argument("--long-side", type=int, default=2000, help="working resolution")
    ap.add_argument("--save-masks", action="store_true", help="also save 16-bit instance label PNGs")
    args = ap.parse_args()

    exts = (".jpg", ".jpeg", ".png")
    paths = sorted(p for p in glob.glob(os.path.join(args.input, "**", "*"), recursive=True)
                   if p.lower().endswith(exts))
    if not paths:
        raise SystemExit(f"No images found under '{os.path.abspath(args.input)}'. "
                         "Make sure the dataset zips are extracted inside this folder.")
    print(f"Found {len(paths)} images")
    os.makedirs(os.path.join(args.output, "overlays"), exist_ok=True)
    if args.save_masks: os.makedirs(os.path.join(args.output, "masks"), exist_ok=True)
    gt = load_gt(args.gt)
    rows, accs = [], []
    for p in paths:
        t = time.time()
        name = os.path.basename(p)
        im, inst, kinds = count_image(p, args.long_side)
        n = len(inst)
        vis = overlay(im, inst)
        small = cv2.resize(vis, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(args.output, "overlays", name), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if args.save_masks:
            lab = np.zeros(im.shape[:2], np.uint16)
            for k, m in enumerate(inst, 1): lab[m] = k
            cv2.imwrite(os.path.join(args.output, "masks", os.path.splitext(name)[0] + ".png"), lab)
        g = gt.get(name)
        acc = None if g is None else max(0.0, 1 - abs(n - g) / g)
        if acc is not None: accs.append(acc)
        rows.append(dict(image=name, predicted=n, nuts=kinds.count("nut"),
                         gt_count="" if g is None else g,
                         accuracy="" if acc is None else f"{100*acc:.2f}",
                         seconds=f"{time.time()-t:.1f}"))
        print(f"{name:30s} pred={n:4d}" + ("" if g is None else f"  gt={g:4d}  acc={100*acc:6.2f}%"))
    with open(os.path.join(args.output, "counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if accs:
        print(f"\nMean count accuracy over {len(accs)} images with GT: {100*np.mean(accs):.2f}%")

if __name__ == "__main__":
    main()
