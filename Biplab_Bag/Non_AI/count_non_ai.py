"""Classical (non-AI) object counting + instance-mask overlay.

No learning of any kind is used - only image processing and geometry.

Pipeline
--------
1. Resize to a working resolution (long side 2000 px).
2. Illumination normalisation: the background is estimated with a large
   grey-scale closing (objects are darker than the backdrop) and subtracted,
   which removes vignetting and light gradients.
3. Foreground score = background-relative darkness + colour distance to the
   backdrop colour (brass screws are bright but coloured); Otsu threshold.
4. Connected components.  Each component is a single item or a cluster of
   touching / overlapping items.
5. Item size ("unit area") is estimated per image with a quantisation fit:
   the largest area u for which the component areas are close to integer
   multiples of u.  It works even when hardly any item lies isolated.
6. Two counting rules, chosen automatically per image:
   * hollow items (nuts / washers): every round through-hole is one item.
     Holes are told apart from the gaps between touching nuts by size and
     circularity.  Hole centres become watershed markers.
   * solid items (screws / bolts): items in a component =
     round(component area / unit area).  Masks are split with a
     marker-controlled watershed seeded at the k thickest points (heads).
7. Overlay: coloured instance masks, outlines, ids and the total count.

Usage
-----
    python count_non_ai.py -i <images or folders> -o <out_dir> [--save-masks]
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

WORK_SIDE = 2000          # working resolution (long side, px)
MIN_AREA_FRAC = 2e-5      # components smaller than this fraction of the image are noise
MIN_ITEM_FRAC = 0.45      # isolated components smaller than this x unit area are fragments
HOLE_AREA = (0.65, 1.45)  # accepted hole area, relative to the median hole
HOLE_CIRC = 0.85          # accepted hole circularity (gaps between nuts are irregular)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def list_images(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() in IMG_EXT:
            out.append(p)
    return out


def resize_long_side(img: np.ndarray, side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img, s


def disk(r: float) -> np.ndarray:
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# --------------------------------------------------------------------------- #
# 1) Foreground segmentation
# --------------------------------------------------------------------------- #
def foreground_score(img: np.ndarray) -> np.ndarray:
    """Per-pixel objectness (float32, ~0 on the backdrop)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    # background luminance: a closing removes the (dark) objects; computed at
    # quarter scale for speed, then smoothed and upsampled
    small = cv2.resize(L, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, disk(12))
    bg = cv2.GaussianBlur(bg, (0, 0), 8)
    bg = cv2.resize(bg, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR)
    dark = np.clip(bg - L, 0, None)
    ab = lab[..., 1:] - 128.0
    bg_ab = np.median(ab.reshape(-1, 2), axis=0)
    chroma = np.linalg.norm(ab - bg_ab, axis=2)
    return dark + chroma


def segment(img: np.ndarray) -> np.ndarray:
    s8 = np.clip(foreground_score(img) * 3.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (s8 > max(t, 45)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(1))
    return mask


def find_holes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(filled mask, holes) - holes = background not connected to the border."""
    inv = (1 - mask).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    border = np.zeros(n, bool)
    border[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = True
    holes = (~border[lab]) & (inv > 0)
    return (mask | holes).astype(np.uint8), holes.astype(np.uint8)


# --------------------------------------------------------------------------- #
# 2) Item-size estimation
# --------------------------------------------------------------------------- #
def estimate_unit_area(areas: np.ndarray) -> float:
    """Largest u such that the component areas are ~integer multiples of u.

    A plain median fails when most items touch (the median component is a
    cluster) or when there are only a handful of items.  For the true u the
    ratios a/u lie close to integers; for u/2, u/3 ... they do too, so we
    keep the *largest* well-fitting u.
    """
    a = np.sort(areas[areas > 0].astype(float))
    if len(a) == 0:
        return 1.0
    if len(a) < 3:
        return float(a[0])
    med = float(np.median(a))
    us = np.geomspace(0.25 * med, 1.6 * med, 300)
    scores = np.empty_like(us)
    for j, u in enumerate(us):
        r = a / u
        r = r[r <= 6.5]                        # big clusters carry little information
        if len(r) < max(3, 0.3 * len(a)):
            scores[j] = 1.0
            continue
        k = np.maximum(1, np.round(r))
        scores[j] = np.mean(np.abs(r - k) / np.sqrt(k))
    best = scores.min()
    good = us[scores <= best + 0.04]
    u = float(good.max())
    # refine: median of per-item areas of 1-3 item components
    k = np.maximum(1, np.round(a / u))
    sel = (k <= 3) & (np.abs(a / u - k) < 0.3)
    if sel.sum() >= 3:
        u = float(np.median(a[sel] / k[sel]))
    return u


# --------------------------------------------------------------------------- #
# 3) Instance splitting
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    unit_area: float = 0.0
    hollow: bool = False
    count: int = 0


def hole_table(holes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Label holes and return (labels, area, circularity, centroids)."""
    n, hlab, hst, hcen = cv2.connectedComponentsWithStats(holes, connectivity=4)
    area = hst[:, cv2.CC_STAT_AREA].astype(float)
    circ = np.zeros(n)
    for i in range(1, n):
        x, y, w, h, a = hst[i]
        sub = (hlab[y:y + h, x:x + w] == i).astype(np.uint8)
        cs, _ = cv2.findContours(np.pad(sub, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        p = cv2.arcLength(max(cs, key=len), True)
        circ[i] = 4 * np.pi * a / max(p, 1.0) ** 2
    return hlab, area, circ, hcen


def kmeans_markers(comp: np.ndarray, k: int) -> np.ndarray:
    """Fallback seeds: k-means on pixel coordinates, snapped onto the object."""
    ys, xs = np.nonzero(comp)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    k = max(1, min(k, len(pts)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, _, centers = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    m = np.zeros(comp.shape, np.int32)
    for j, c in enumerate(centers, 1):
        q = int(np.argmin(((pts - c) ** 2).sum(1)))
        m[int(pts[q, 1]), int(pts[q, 0])] = j
    return m


def split_instances(mask: np.ndarray) -> tuple[np.ndarray, Stats]:
    st = Stats()
    H, W = mask.shape
    filled, holes = find_holes(mask)
    n, lab, cst, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, np.int32), st
    # object material per component (holes and enclosed gaps excluded)
    material = np.bincount(lab[mask > 0].ravel(), minlength=n).astype(float)

    valid = material >= MIN_AREA_FRAC * mask.size
    valid[0] = False
    # thin strips glued to the image border are lighting artefacts
    for i in np.nonzero(valid)[0]:
        x, y, w, h, _ = cst[i]
        if (x == 0 or y == 0 or x + w == W or y + h == H) and min(w, h) < 0.012 * max(H, W):
            valid[i] = False
    if not valid.any():
        return np.zeros(mask.shape, np.int32), st

    # ---- hollow-item test: do typical components contain one round hole?
    hlab, harea, hcirc, hcen = hole_table(holes)
    owner = lab[hcen[:, 1].astype(int).clip(0, H - 1), hcen[:, 0].astype(int).clip(0, W - 1)]
    owner[0] = 0
    cand = (harea > 0.02 * np.median(material[valid])) & (hcirc > HOLE_CIRC)
    cand[0] = False
    hole_ok = np.zeros(len(harea), bool)
    if cand.any():
        med_hole = float(np.median(harea[cand]))
        hole_ok = cand & (harea > HOLE_AREA[0] * med_hole) & (harea < HOLE_AREA[1] * med_hole)
    n_hole = np.bincount(owner[hole_ok], minlength=n)
    one_hole = valid & (n_hole == 1)
    # nut images: ~100 % of components hold a round hole; screw images: < 5 %
    st.hollow = bool((valid & (n_hole >= 1)).sum() >= 0.5 * valid.sum() and one_hole.sum() >= 3)

    unit = float(np.median(material[one_hole])) if st.hollow else estimate_unit_area(material[valid])
    st.unit_area = unit
    # drop small isolated fragments (dust, shadows, vignetting corners)
    valid &= material >= MIN_ITEM_FRAC * unit

    out = np.zeros(mask.shape, np.int32)
    nid = 0
    pad = 3
    r_item = np.sqrt(unit / np.pi)
    for i in np.nonzero(valid)[0]:
        x, y, w, h, _ = cst[i]
        y0, y1 = max(0, y - pad), min(H, y + h + pad)
        x0, x1 = max(0, x - pad), min(W, x + w + pad)
        comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)

        if st.hollow:
            # one accepted through-hole == one nut
            hl = hlab[y0:y1, x0:x1] * comp
            ids = [j for j in np.unique(hl) if j > 0 and hole_ok[j]]
            markers = np.zeros(comp.shape, np.int32)
            for m_i, j in enumerate(ids, 1):
                markers[hl == j] = m_i
            if not ids:                       # nut with hidden hole / standing on edge
                markers[comp > 0] = 1
            k = max(1, len(ids))
        else:
            # overlapping / touching solid items: count from the area
            k = max(1, int(np.round(material[i] / unit)))
            if k == 1:
                markers = comp.astype(np.int32)
            else:
                pk = peak_local_max(dist, min_distance=max(2, int(0.6 * r_item)), labels=comp.astype(np.int32),
                                    num_peaks=k, exclude_border=False)
                if len(pk) == k:
                    markers = np.zeros(comp.shape, np.int32)
                    for m_i, (py, px) in enumerate(pk, 1):
                        markers[py, px] = m_i
                else:
                    markers = kmeans_markers(comp, k)
        if st.hollow and markers.max() > 1:
            # nuts: grow every hole outwards (nearest-hole partition), keep only
            # nut material + the nut's own hole (gaps between nuts stay empty)
            relief = cv2.distanceTransform((markers == 0).astype(np.uint8), cv2.DIST_L2, 5)
            keep = (mask[y0:y1, x0:x1] > 0) | (markers > 0)
            seg = watershed(relief, markers=markers, mask=(comp > 0) & keep)
        elif markers.max() > 1:
            seg = watershed(-dist, markers=markers, mask=comp > 0)
        else:
            seg = comp.astype(np.int32)
        sub = out[y0:y1, x0:x1]
        labels_here = [j for j in np.unique(seg) if j > 0]
        for j in labels_here:
            nid += 1
            sub[seg == j] = nid
        nid += max(0, k - len(labels_here))  # keep the count consistent if a seed got lost
    st.count = nid
    return out, st


# --------------------------------------------------------------------------- #
# 4) Visualisation
# --------------------------------------------------------------------------- #
def overlay(img: np.ndarray, labels: np.ndarray, count: int, title: str, method: str = "Non-AI") -> np.ndarray:
    n = int(labels.max())
    rng = np.random.default_rng(7)
    colors = rng.integers(40, 255, size=(n + 1, 3), dtype=np.uint8)
    colors[0] = 0
    fg = labels > 0
    out = img.copy()
    out[fg] = (0.45 * img[fg] + 0.55 * colors[labels][fg]).astype(np.uint8)
    edges = (cv2.morphologyEx(labels.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3))) > 0) & fg
    out[edges] = (0, 0, 0)
    fs = max(0.3, img.shape[1] / 4500)
    for j, (cy, cx) in enumerate(ndi.center_of_mass(fg, labels, range(1, n + 1)), 1):
        if np.isfinite(cx):
            cv2.putText(out, str(j), (int(cx) - 6, int(cy) + 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), 1, cv2.LINE_AA)
    sc = img.shape[1] / 1400
    cv2.rectangle(out, (0, 0), (img.shape[1], int(60 * sc)), (255, 255, 255), -1)
    cv2.putText(out, f"{title}  |  {method} count = {count}", (int(15 * sc), int(42 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1 * sc, (0, 0, 200), max(1, int(2 * sc)), cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def process(path: Path) -> tuple[int, np.ndarray, np.ndarray, Stats]:
    img0 = cv2.imread(str(path))
    if img0 is None:
        raise IOError(f"cannot read {path}")
    img, _ = resize_long_side(img0, WORK_SIDE)
    labels, st = split_instances(segment(img))
    return st.count, img, labels, st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", nargs="+", required=True, help="image files and/or folders")
    ap.add_argument("--output", "-o", default="outputs", help="output folder")
    ap.add_argument("--save-masks", action="store_true", help="also save 16-bit instance label PNGs")
    args = ap.parse_args()

    out_dir = Path(args.output)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    images = list_images(args.input)
    if not images:
        raise SystemExit("no images found")
    rows = []
    for p in images:
        t0 = time.time()
        count, img, labels, st = process(p)
        dt = time.time() - t0
        cv2.imwrite(str(out_dir / "overlays" / f"{p.stem}_overlay.jpg"), overlay(img, labels, count, p.name),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        if args.save_masks:
            cv2.imwrite(str(out_dir / "masks" / f"{p.stem}_labels.png"), labels.astype(np.uint16))
        rows.append({"image": p.name, "count": count, "mode": "hollow" if st.hollow else "solid",
                     "unit_area_px": round(st.unit_area, 1), "seconds": round(dt, 2)})
        print(f"{p.name:28s} count={count:4d}  mode={rows[-1]['mode']:6s} unit={st.unit_area:7.1f}px  {dt:.2f}s")
    with open(out_dir / "counts.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nSaved {len(rows)} results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
