"""Build the YOLO-seg training set (one class, ``item``) from automatic labels.

Labels come from the ground-truth step (``../GroundTruth``), not from hand drawing:
    * objects both counters agree on (Non-AI and SAM 2, matched one-to-one by mask IoU)
      get the SAM 2 mask, or the union of both masks when SAM 2 outlined only part of
      the object (a screw head without its thread);
    * disputed objects accepted on review get the instance under their stored point.

Images are resized like the counters (long side 2000 px) and cut into 640 px tiles with
160 px overlap. Each object becomes a polygon in every tile holding at least 15% of it.
Most empty tiles are dropped. Held-out test images are skipped.

Needs ``../GroundTruth/sam/*.npz`` (run ``sam_count.py`` there first).

Usage:
    uv run make_dataset.py /path/to/data
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from splits import split_of

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Non_AI"))
sys.path.insert(0, str(ROOT / "GroundTruth"))
import count as nonai  # noqa: E402  (Non_AI/count.py: shadows AI/count.py, which is not used here)
from compare import match  # noqa: E402

TILE, STEP = 640, 480
DATASET = HERE / "datasets" / "items"


def instance_masks(path: Path, accepted: list[dict]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the resized image and one boolean mask per labelled object."""
    res = nonai.count_image(path)
    a = res.labels
    b = np.load(ROOT / "GroundTruth" / "sam" / f"{path.parent.name}__{path.stem}.npz")["labels"]
    pairs, _, _ = match(a, b)
    masks = []
    for i, j in pairs:
        ma, mb = a == i, b == j
        masks.append(mb if mb.sum() >= 0.7 * ma.sum() else ma | mb)

    paired_a, paired_b = {i for i, _ in pairs}, {j for _, j in pairs}
    for cand in accepted:
        x, y = (round(v) for v in cand["point"])
        if a[y, x] and a[y, x] not in paired_a:
            masks.append(a == a[y, x])
        elif b[y, x] and b[y, x] not in paired_b:
            masks.append(b == b[y, x])
    return res.image, masks


def tile_starts(n: int) -> list[int]:
    """Return tile start offsets covering ``n`` pixels."""
    starts = list(range(0, max(n - TILE, 0) + 1, STEP))
    if starts[-1] + TILE < n:
        starts.append(n - TILE)
    return starts


def polygon_line(crop: np.ndarray) -> str | None:
    """Return a YOLO-seg label line for the largest outline in ``crop``, or None."""
    contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    if len(c) < 3:
        return None
    pts = c.reshape(-1, 2).astype(np.float32) / TILE
    return "0 " + " ".join(f"{v:.5f}" for v in pts.ravel())


def write_tiles(img: np.ndarray, masks: list[np.ndarray], out: Path, name: str) -> int:
    """Write image tiles and label files for one image; return the number of tiles written."""
    h, w = img.shape[:2]
    areas = [m.sum() for m in masks]
    written = 0
    for y in tile_starts(h):
        for x in tile_starts(w):
            lines = []
            for m, full in zip(masks, areas, strict=True):
                crop = m[y : y + TILE, x : x + TILE]
                part = crop.sum()
                if part and part >= 0.15 * full and (line := polygon_line(crop)):
                    lines.append(line)
            if not lines and np.random.default_rng(x * 7 + y).random() > 0.15:
                continue  # keep about 15% of the empty tiles
            stem = f"{name}_{y}_{x}"
            cv2.imwrite(
                str(out / "images" / f"{stem}.jpg"),
                img[y : y + TILE, x : x + TILE],
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines))
            written += 1
    return written


def main(data: Path) -> None:
    """Build ``datasets/items`` with train and val splits and its ``data.yaml``."""
    review = json.loads((ROOT / "GroundTruth" / "review.json").read_text())
    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ("train", "val"):
        (DATASET / split / "images").mkdir(parents=True)
        (DATASET / split / "labels").mkdir(parents=True)

    for f in sorted(data.rglob("*.jpg")):
        key, split = f"{f.parent.name}/{f.name}", split_of(f.name)
        if split == "test":
            print(f"test  {key} (held out)")
            continue
        img, masks = instance_masks(f, review[key]["accepted"])
        n = write_tiles(img, masks, DATASET / split, f"{f.parent.name}__{f.stem}")
        print(f"{split:5s} {key}: {len(masks)} objects, {n} tiles", flush=True)
    (DATASET / "data.yaml").write_text(
        f"path: {DATASET}\ntrain: train/images\nval: val/images\nnames:\n  0: item\n"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
