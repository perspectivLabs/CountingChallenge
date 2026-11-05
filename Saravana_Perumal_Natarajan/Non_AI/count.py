"""Count screws, bolts and nuts with classical image processing (no machine learning).

Pipeline:
    1. Load the image (EXIF orientation applied) and resize it to a long side of 2000 px.
    2. Remove uneven lighting with a polynomial background fit, threshold with Otsu and
       mask dark lens-vignette corners.
    3. Find connected components; the most common component area is one object.
    4. Decide the image type: nuts if isolated objects have exactly one hole each.
    5. Count and build one mask per object:
         * nuts: one nut per hole, pixels assigned to the nearest hole (Voronoi split);
         * screws: round objects found as distance-transform peaks, the rest of a
           touching group divided by the area of one screw and split with k-means.

Usage:
    uv run count.py /path/to/images_or_folders -o outputs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

WORK_SIZE = 2000


@dataclass
class Result:
    """Counting result for one image.

    Attributes:
        path: Source image path.
        count: Number of objects found.
        kind: ``"nut"`` or ``"screw"``.
        unit_area: Area of one object in pixels (working resolution).
        labels: Instance label image, 0 = background.
        image: Resized BGR image that ``labels`` refers to.
    """

    path: str
    count: int
    kind: str
    unit_area: float
    labels: np.ndarray = field(repr=False)
    image: np.ndarray = field(repr=False)


def load(path: str | Path, work_size: int = WORK_SIZE) -> np.ndarray:
    """Read an image with EXIF orientation applied and shrink its long side to ``work_size``."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    scale = work_size / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


# Foreground segmentation


def _fit_background(gray: np.ndarray, bg_mask: np.ndarray, degree: int = 4) -> np.ndarray:
    """Fit a 2-D polynomial surface to the pixels in ``bg_mask`` and evaluate it everywhere."""
    h, w = gray.shape
    ys, xs = np.nonzero(bg_mask)
    if len(xs) > 40000:
        idx = np.random.default_rng(0).choice(len(xs), 40000, replace=False)
        xs, ys = xs[idx], ys[idx]
    xn, yn = xs / w * 2 - 1, ys / h * 2 - 1
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    design = np.stack([xn**i * yn**j for i, j in terms], axis=1)
    coef, *_ = np.linalg.lstsq(design, gray[ys, xs].astype(np.float64), rcond=None)
    gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    return sum(c * gx**i * gy**j for c, (i, j) in zip(coef, terms, strict=True))


def _vignette(gray: np.ndarray) -> np.ndarray:
    """Mask of dark lens-vignette corners: large, smooth dark areas touching the border."""
    blur = cv2.GaussianBlur(gray, (0, 0), 25)  # wipes out objects, keeps broad shading
    dark = (blur < 0.8 * np.median(gray)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    h, w = gray.shape
    out = np.zeros(gray.shape, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        on_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if on_border and area > 0.01 * h * w:
            out |= lab == i
    return cv2.dilate(out.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0


def foreground(img: np.ndarray) -> np.ndarray:
    """Return a uint8 mask (0/255) of the objects; holes inside nuts stay background."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rough = cv2.medianBlur(gray.astype(np.uint8), 151).astype(np.float32)
    bg_mask = cv2.erode(((rough - gray) < 15).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    for _ in range(2):  # fit, re-classify, fit again
        diff = np.clip(_fit_background(gray, bg_mask) - gray, 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg_mask = cv2.erode((diff < t * 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Brass screws are saturated even where they are not much darker than the paper.
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    saturated = (sat - np.median(sat[bg_mask])) > 45

    mask = ((diff > t * 0.8) | saturated).astype(np.uint8) * 255
    mask[_vignette(gray)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


# Counting


def _mode_area(areas: np.ndarray) -> float:
    """Return the most common area (peak of the log-area histogram), i.e. one object."""
    la = np.log(areas)
    hist, edges = np.histogram(la, bins=40)
    k = np.argmax(np.convolve(hist, [1, 2, 3, 2, 1], mode="same"))
    lo, hi = edges[max(k - 2, 0)], edges[min(k + 3, len(edges) - 1)]
    return float(np.exp(np.median(la[(la >= lo) & (la <= hi)])))


def _holes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split background into enclosed holes; return ``(mask with holes filled, holes)``."""
    flood = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(flood[1:-1, 1:-1])
    return cv2.bitwise_or(mask, holes), holes


def count_image(path: str | Path) -> Result:
    """Count the objects in one image and build an instance mask for each."""
    img = load(path)
    mask = foreground(img)
    filled, holes = _holes(mask)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    keep = areas > 40  # dust
    unit = _mode_area(areas[keep])
    keep &= areas > 0.25 * unit

    # Shading left at the image border: large blobs on an edge, or anything in a corner.
    h, w = filled.shape
    x, y, bw, bh = (stats[1:, i] for i in range(4))
    edges_hit = (x == 0).astype(int) + (y == 0) + (x + bw >= w) + (y + bh >= h)
    keep &= ~((edges_hit > 0) & (areas > 2.5 * unit)) & (edges_hit < 2)

    hn, _, hstats, hcent = cv2.connectedComponentsWithStats(holes, connectivity=4)
    hareas = hstats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    hole_unit = _mode_area(hareas[hareas > 5]) if (hareas > 5).sum() >= 5 else 0.0
    good_hole = (hareas > 0.3 * hole_unit) & (hareas < 3.0 * hole_unit)
    hole_owner = lab[hcent[1:, 1].astype(int), hcent[1:, 0].astype(int)] if hn > 1 else np.zeros(0, int)

    # Nut image if most isolated objects have exactly one hole.
    singles = [i for i in range(1, n) if keep[i - 1] and 0.6 * unit < areas[i - 1] < 1.5 * unit]
    holes_per_single = [int(np.sum((hole_owner == i) & good_hole)) for i in singles]
    is_nut = bool(singles) and np.mean(np.array(holes_per_single) == 1) > 0.7

    labels = np.zeros(lab.shape, np.int32)
    if is_nut:
        count = _label_nuts(labels, lab, keep, areas, unit, _nut_holes(mask, hole_unit), mask)
    else:
        count = _label_screws(labels, lab, keep, areas, unit)
    return Result(str(path), count, "nut" if is_nut else "screw", unit, labels, img)


# Instance masks


def _bbox(comp: np.ndarray, pad: int = 0) -> tuple[slice, slice]:
    """Return the bounding box of a boolean mask as a pair of slices."""
    ys, xs = np.nonzero(comp)
    h, w = comp.shape
    return (
        slice(max(ys.min() - pad, 0), min(ys.max() + 1 + pad, h)),
        slice(max(xs.min() - pad, 0), min(xs.max() + 1 + pad, w)),
    )


def _circularity(comp: np.ndarray) -> float:
    """Return 4*pi*area/perimeter^2 of the largest outline in ``comp`` (1.0 for a disc)."""
    contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    p = cv2.arcLength(c, True)
    return 4 * np.pi * cv2.contourArea(c) / p**2 if p else 0.0


def _split_kmeans(comp: np.ndarray, k: int) -> np.ndarray:
    """Split a mask into ``k`` parts by k-means on pixel coordinates; return labels 1..k."""
    out = np.zeros(comp.shape, np.int32)
    if k <= 1:
        out[comp] = 1
        return out
    ys, xs = np.nonzero(comp)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    cv2.setRNGSeed(0)
    _, lbl, _ = cv2.kmeans(pts, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    out[ys, xs] = lbl.ravel() + 1
    return out


def _paint(labels: np.ndarray, sl: tuple[slice, slice], seg: np.ndarray, next_id: int) -> int:
    """Copy the non-empty segments of ``seg`` into ``labels[sl]`` as new ids; return the next free id."""
    for s in range(1, seg.max() + 1):
        m = seg == s
        if m.any():
            labels[sl][m] = next_id
            next_id += 1
    return next_id


def _label_screws(labels, lab, keep, areas, unit) -> int:
    """Label lying screws and round objects (bolt heads, upright screws); return the count.

    Inside a touching group, round objects show up as distance-transform peaks close to
    their radius (a lying screw is too thin). Each peak becomes a disc; the rest of the
    group is divided into whole screws.
    """
    ids = np.nonzero(keep)[0] + 1
    boxes = {i: _bbox(lab == i, pad=2) for i in ids}
    round_ids = {
        i for i in ids if _circularity(lab[boxes[i]] == i) > 0.8 and 0.3 * unit < areas[i - 1] < 3 * unit
    }
    r_c = np.sqrt(np.median([areas[i - 1] for i in round_ids]) / np.pi) if len(round_ids) >= 3 else 0.0
    flat = np.array([areas[i - 1] for i in ids if i not in round_ids])
    screw_area = _mode_area(flat) if len(flat) >= 3 else unit

    next_id = 1
    for i in ids:
        sl = boxes[i]
        comp = lab[sl] == i
        seg = np.zeros(comp.shape, np.int32)
        k = 0
        if i in round_ids:
            seg[comp] = k = 1
        elif r_c:
            dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
            yy, xx = np.indices(comp.shape)
            while dt.max() >= 0.8 * r_c:
                cy, cx = np.unravel_index(dt.argmax(), dt.shape)
                d2 = (yy - cy) ** 2 + (xx - cx) ** 2
                k += 1
                seg[comp & (seg == 0) & (d2 <= (1.1 * r_c) ** 2)] = k
                dt[d2 <= (1.6 * r_c) ** 2] = 0
        rest = comp & (seg == 0)
        # The outline and shadow of a touching group add area, so a further screw
        # needs 0.65 of a screw's area instead of 0.5.
        n_screws = int(np.floor(rest.sum() / screw_area + 0.35))
        if k == 0:
            n_screws = max(n_screws, 1)
        if n_screws:
            sub = _split_kmeans(rest, n_screws)
            seg[sub > 0] = sub[sub > 0] + k
        next_id = _paint(labels, sl, seg, next_id)
    return next_id - 1


def _nut_holes(mask: np.ndarray, hole_unit: float) -> np.ndarray:
    """Return the ``(x, y)`` centres of nut holes.

    A closing with a disc slightly larger than a hole fills the hole even when glare
    breaks the nut's dark ring, so ``closed - mask`` holds the holes. Gaps between
    touching nuts are rejected by size and circularity.
    """
    d = int(2 * np.sqrt(hole_unit / np.pi) * 1.8) | 1
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    cand = cv2.morphologyEx(
        cv2.bitwise_and(closed, cv2.bitwise_not(mask)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    centres = []
    for c in contours:
        a, p = cv2.contourArea(c), cv2.arcLength(c, True)
        if 0.35 * hole_unit < a < 2.2 * hole_unit and p > 0 and 4 * np.pi * a / p**2 >= 0.6:
            m = cv2.moments(c)
            centres.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    return np.array(centres, np.float32).reshape(-1, 2)


def _label_nuts(labels, lab, keep, areas, unit, centres, fg) -> int:
    """Label one nut per hole; return the count.

    Each pixel goes to the nearest hole centre within one nut radius (Voronoi split).
    A square gap between four touching nuts can pass as a hole, but its cell only
    collects slivers of the neighbours, so the weakest cell under 0.7 nut is dropped
    and the split recomputed, one at a time. Leftover solid pieces of at least half a
    nut are nuts whose hole is hidden (lying on another nut or standing on edge).
    """
    radius = np.sqrt(unit / 2.6) * 1.15  # hexagon area = 2.6 * R^2
    owner = lab[centres[:, 1].astype(int), centres[:, 0].astype(int)] if len(centres) else np.zeros(0, int)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    next_id = 1
    for i in np.nonzero(keep)[0] + 1:
        sl = _bbox(lab == i)
        comp = lab[sl] == i
        solid = comp & (fg[sl] > 0)
        ys, xs = np.nonzero(comp)
        cs = (centres[owner == i] - [sl[1].start, sl[0].start]).astype(np.float32)
        seg = np.zeros(comp.shape, np.int32)
        while len(cs):
            seg[:] = 0
            d = np.hypot(xs[:, None] - cs[None, :, 0], ys[:, None] - cs[None, :, 1])
            near = d.argmin(1)
            dn = d[np.arange(len(near)), near]
            inside = np.where(solid[ys, xs], dn < radius, dn < 0.45 * radius)  # body + own hole
            seg[ys[inside], xs[inside]] = near[inside] + 1
            body = np.bincount(seg[solid], minlength=len(cs) + 1)[1:]
            weakest = int(body.argmin())
            if body[weakest] >= 0.7 * unit:
                break
            cs = np.delete(cs, weakest, axis=0)

        if len(cs):
            rest = cv2.morphologyEx((solid & (seg == 0)).astype(np.uint8), cv2.MORPH_OPEN, open_k)
            n, plab, pstats, _ = cv2.connectedComponentsWithStats(rest)
            for j in range(1, n):
                a = pstats[j, cv2.CC_STAT_AREA]
                if a >= 0.5 * unit:
                    sub = _split_kmeans(plab == j, max(1, int(round(a / unit))))
                    seg[sub > 0] = sub[sub > 0] + seg.max()
        else:  # no visible hole at all
            seg = _split_kmeans(comp, max(1, int(round(areas[i - 1] / unit))))
        next_id = _paint(labels, sl, seg, next_id)
    return next_id - 1


# Output


def overlay(res: Result, alpha: float = 0.55) -> np.ndarray:
    """Draw every instance in its own colour with a dark outline and the count in the corner."""
    img = res.image.copy()
    colours = np.random.default_rng(1).integers(60, 255, size=(res.labels.max() + 1, 3)).astype(np.uint8)
    m = res.labels > 0
    img[m] = (img[m] * (1 - alpha) + colours[res.labels][m] * alpha).astype(np.uint8)
    edges = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    edges |= cv2.Laplacian(res.labels.astype(np.float32), cv2.CV_32F) != 0
    img[edges & m] = (20, 20, 20)
    text, scale = f"count: {res.count}", img.shape[1] / 900
    org = (20, int(50 * scale))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), int(6 * scale))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 200), int(2.5 * scale))
    return img


def image_files(inputs: list[str]) -> list[Path]:
    """Expand files and folders (searched recursively) into a sorted list of images."""
    files: list[Path] = []
    for p in map(Path, inputs):
        files += sorted(p.rglob("*.jpg")) + sorted(p.rglob("*.png")) if p.is_dir() else [p]
    return files


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description="Count objects with classical image processing.")
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("-o", "--out", default="outputs", help="folder for overlays and counts.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for f in image_files(args.inputs):
        res = count_image(f)
        cv2.imwrite(str(out / f"{f.parent.name}__{f.stem}.jpg"), overlay(res), [cv2.IMWRITE_JPEG_QUALITY, 90])
        counts[f"{f.parent.name}/{f.name}"] = res.count
        print(f"{f.parent.name}/{f.name:28s} {res.kind:6s} {res.count}")
    (out / "counts.json").write_text(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
