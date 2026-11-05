"""Second, independent counter for building ground truth: SAM 2 with point prompts on tiles.

SAM 2 is a general segmenter that knows nothing about screws or nuts, so its mistakes
are unrelated to those of the classical Non-AI pipeline.

Per image:
    1. Resize like the Non-AI counter (EXIF orientation, long side 2000 px) so the
       coordinates of both methods line up.
    2. Measure the area of one object on isolated objects (``estimate_unit``).
    3. Cut into 512 px tiles with 128 px overlap and skip plain-background tiles.
    4. Prompt SAM 2 with points on dark pixels; per prompt keep the largest candidate
       mask of single-object size (``tile_masks``).
    5. Keep a mask only if its centre lies in the tile's core, so objects in the
       overlap are counted once.

Writes ``sam/<folder>__<name>.npz`` holding an int32 instance label image.

Usage:
    uv run sam_count.py /path/to/data [--only NAME ...] [--force]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import SAM

HERE = Path(__file__).resolve().parent
WORK_SIZE = 2000
TILE, OVERLAP = 512, 128
DARK = 30  # grey levels below the paper that count as "object"


def load(path: Path) -> np.ndarray:
    """Read an image (EXIF orientation applied) and shrink its long side to ``WORK_SIZE``."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    s = WORK_SIZE / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def tiles(h: int, w: int) -> Iterator[tuple[int, int]]:
    """Yield the ``(y, x)`` corners of overlapping tiles covering an ``h`` x ``w`` image."""
    step = TILE - OVERLAP
    ys = list(range(0, max(h - TILE, 0) + 1, step))
    xs = list(range(0, max(w - TILE, 0) + 1, step))
    if ys[-1] + TILE < h:
        ys.append(h - TILE)
    if xs[-1] + TILE < w:
        xs.append(w - TILE)
    for y in ys:
        for x in xs:
            yield y, x


def darkness(img: np.ndarray) -> np.ndarray:
    """Return how much darker each pixel is than the paper.

    The paper level is the brightest value over a window wider than the biggest pile
    of objects, smoothed. A median would fail inside dense piles, where objects are
    the majority.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    small = cv2.resize(gray, (w // 16, h // 16), interpolation=cv2.INTER_AREA)
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
    bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 10)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR) - gray.astype(np.float32)


def build_predictor(device: str):
    """Load SAM 2 (``weights/sam2.1_b.pt``, downloaded on first use) and return its predictor."""
    model = SAM(str(HERE / "weights" / "sam2.1_b.pt"))
    model(np.zeros((64, 64, 3), np.uint8), points=[[[8, 8]]], labels=[[1]], device=device, verbose=False)
    model.predictor.args.conf = 0.0  # keep every candidate so masks stay aligned with their prompts
    return model.predictor


def _dark_points(t_dark: np.ndarray, step: int = 10) -> np.ndarray:
    """Return grid points (x, y) on dark pixels, in a fixed random order."""
    th, tw = t_dark.shape
    gy, gx = np.mgrid[step // 2 : th : step, step // 2 : tw : step]
    on = t_dark[gy, gx] > DARK
    pts = np.stack([gx[on], gy[on]], 1)
    return pts[np.random.default_rng(0).permutation(len(pts))]


def _prompt(pred, pts: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Run SAM 2 on point prompts; return masks ``(n, 3, h, w)`` and scores ``(n, 3)``."""
    r = pred(points=[[p.tolist()] for p in pts], labels=[[1]] * len(pts), multimask_output=True)[0]
    masks = (r.masks.data.cpu().numpy() > 0.5).reshape(len(pts), 3, *shape)
    return masks, r.boxes.conf.cpu().numpy().reshape(len(pts), 3)


def estimate_unit(pred, img: np.ndarray, dark: np.ndarray, want: int = 25) -> float:
    """Return the area of one object, measured on isolated objects only.

    For an object with only paper around it, SAM's largest candidate is the whole
    object (there is nothing to merge with). Isolated means the ring just outside the
    mask is background. The busiest tiles are visited first.
    """
    ring_kernel = np.ones((9, 9), np.uint8)
    areas: list[int] = []
    busiest = sorted(
        tiles(*img.shape[:2]), key=lambda t: -(dark[t[0] : t[0] + TILE, t[1] : t[1] + TILE] > DARK).mean()
    )
    for y, x in busiest:
        t_dark = dark[y : y + TILE, x : x + TILE]
        pts = _dark_points(t_dark, step=14)[:64]
        if not len(pts):
            break
        pred.set_image(img[y : y + TILE, x : x + TILE])
        masks, conf = _prompt(pred, pts, t_dark.shape)
        pred.reset_image()
        th, tw = t_dark.shape
        seen = np.zeros((th, tw), bool)
        for i in range(len(pts)):
            ok = [k for k in range(3) if conf[i, k] > 0.7 and masks[i, k].sum() > 40]
            if not ok:
                continue
            m = masks[i, max(ok, key=lambda k: masks[i, k].sum())]
            ys, xs = np.nonzero(m)
            if (m & seen).any() or ys.min() == 0 or xs.min() == 0 or ys.max() == th - 1 or xs.max() == tw - 1:
                continue
            ring = cv2.dilate(m.astype(np.uint8), ring_kernel).astype(bool) & ~m
            if m.sum() > 150 and (t_dark[ring] > 40).mean() < 0.15 and t_dark[m].mean() > 40:
                areas.append(int(m.sum()))
                seen |= m
        if len(areas) >= want:
            break
    return float(np.median(areas))


def tile_masks(pred, tile: np.ndarray, t_dark: np.ndarray, unit: float, batch: int = 64) -> list[np.ndarray]:
    """Segment the objects of one tile with point prompts.

    SAM returns three candidates per prompt (part, object, group). The largest one of
    single-object size (0.5 to 1.9 x ``unit``) is used, so a whole screw wins over its
    head alone. After each batch, prompts inside accepted masks are dropped; masks
    mostly covered by earlier ones are parts and are rejected.
    """
    shape = tile.shape[:2]
    pts = _dark_points(t_dark)
    pred.set_image(tile)
    covered = np.zeros(shape, bool)
    kept: list[np.ndarray] = []
    while len(pts):
        b, pts = pts[:batch], pts[batch:]
        masks, conf = _prompt(pred, b, shape)
        area = masks.reshape(len(b), 3, -1).sum(2)
        best = []
        for i in range(len(b)):
            ok = [
                k
                for k in range(3)
                if conf[i, k] > 0.6
                and 0.5 * unit <= area[i, k] <= 1.9 * unit
                and t_dark[masks[i, k]].mean() > 25
            ]
            if ok:
                k = max(ok, key=lambda k: area[i, k])
                best.append((conf[i, k], masks[i, k]))
        for _, m in sorted(best, key=lambda t: -t[0]):
            if (m & covered).sum() <= 0.3 * m.sum():
                kept.append(m)
                covered |= m
        if len(pts):
            pts = pts[~covered[pts[:, 1], pts[:, 0]]]
    pred.reset_image()
    return kept


def segment(pred, img: np.ndarray) -> np.ndarray:
    """Return an instance label image for the whole (resized) image."""
    h, w = img.shape[:2]
    dark = darkness(img)
    unit = estimate_unit(pred, img, dark)
    labels = np.zeros((h, w), np.int32)
    next_id, half = 1, OVERLAP // 2
    for y, x in tiles(h, w):
        t_dark = dark[y : y + TILE, x : x + TILE]
        if (t_dark > DARK).mean() < 0.002:
            continue
        th, tw = t_dark.shape
        cy0, cx0 = (0 if y == 0 else half), (0 if x == 0 else half)
        cy1 = th if y + TILE >= h else th - half
        cx1 = tw if x + TILE >= w else tw - half
        for m in tile_masks(pred, img[y : y + TILE, x : x + TILE], t_dark, unit):
            ys, xs = np.nonzero(m)
            if not (cy0 <= ys.mean() < cy1 and cx0 <= xs.mean() < cx1):
                continue  # centre in the overlap: the neighbouring tile owns it
            region = labels[y : y + th, x : x + tw]
            free = m & (region == 0)
            if free.sum() >= 0.5 * m.sum():
                region[free] = next_id
                next_id += 1
    return labels


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description="SAM 2 tiled counter (ground-truth helper).")
    ap.add_argument("data", help="dataset folder")
    ap.add_argument("--only", nargs="*", help="file names or stems to process")
    ap.add_argument("--force", action="store_true", help="redo images that already have a result")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    out = HERE / "sam"
    out.mkdir(exist_ok=True)
    pred = build_predictor(args.device)
    for f in sorted(Path(args.data).rglob("*.jpg")):
        if args.only and f.name not in args.only and f.stem not in args.only:
            continue
        dst = out / f"{f.parent.name}__{f.stem}.npz"
        if dst.exists() and not args.force:
            print(f"skip {dst.name} (exists)")
            continue
        start = time.time()
        labels = segment(pred, load(f))
        np.savez_compressed(dst, labels=labels)
        print(f"{f.parent.name}/{f.name}: {labels.max()} objects ({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
