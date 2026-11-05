"""Count items with the fine-tuned YOLO11-seg model and draw one mask per item.

The image is resized (long side 2000 px, EXIF orientation applied) and cut into 640 px
tiles with 160 px overlap, the tiling the model was trained on. A detection counts only
if its mask centre lies in the tile's core, so objects in the overlap are counted once.
Masks are painted highest-confidence first; a mask that is mostly covered already is a
duplicate and is dropped.

Usage:
    uv run count.py /path/to/images_or_folders -o outputs [--weights PATH] [--conf 0.35]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
WORK_SIZE, TILE, STEP = 2000, 640, 480
DEFAULT_WEIGHTS = HERE / "model" / "items_yolo11n_seg.pt"


@dataclass
class Result:
    """Counting result for one image.

    Attributes:
        path: Source image path.
        count: Number of items found.
        labels: Instance label image, 0 = background.
        image: Resized BGR image that ``labels`` refers to.
    """

    path: str
    count: int
    labels: np.ndarray = field(repr=False)
    image: np.ndarray = field(repr=False)


def load(path: str | Path) -> np.ndarray:
    """Read an image with EXIF orientation applied and shrink its long side to ``WORK_SIZE``."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    s = WORK_SIZE / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def tile_starts(n: int) -> list[int]:
    """Return tile start offsets covering ``n`` pixels."""
    starts = list(range(0, max(n - TILE, 0) + 1, STEP))
    if starts[-1] + TILE < n:
        starts.append(n - TILE)
    return starts


def default_device() -> str:
    """Pick Apple MPS, CUDA or CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "0" if torch.cuda.is_available() else "cpu"


class Counter:
    """Tiled YOLO11-seg counter.

    Args:
        weights: Model file.
        conf: Detection confidence threshold (0.35 was chosen on the validation images).
        device: Torch device; picked automatically when None.
    """

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, conf: float = 0.35, device: str | None = None):
        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device or default_device()

    def __call__(self, path: str | Path) -> Result:
        """Count the items in one image."""
        img = load(path)
        h, w = img.shape[:2]
        margin = (TILE - STEP) // 2
        detections = []  # (confidence, tile y, tile x, tile-sized mask)
        for y in tile_starts(h):
            for x in tile_starts(w):
                tile = img[y : y + TILE, x : x + TILE]
                th, tw = tile.shape[:2]
                r = self.model.predict(
                    tile,
                    imgsz=TILE,
                    conf=self.conf,
                    iou=0.5,
                    max_det=600,
                    retina_masks=True,
                    device=self.device,
                    verbose=False,
                )[0]
                if r.masks is None:
                    continue
                y0, x0 = (0 if y == 0 else margin), (0 if x == 0 else margin)
                y1 = th if y + TILE >= h else th - margin
                x1 = tw if x + TILE >= w else tw - margin
                masks = r.masks.data.cpu().numpy()[:, :th, :tw] > 0.5
                for m, c in zip(masks, r.boxes.conf.cpu().numpy(), strict=True):
                    ys, xs = np.nonzero(m)
                    if len(xs) >= 20 and y0 <= ys.mean() < y1 and x0 <= xs.mean() < x1:
                        detections.append((float(c), y, x, m))

        labels = np.zeros((h, w), np.int32)
        next_id = 1
        for _, y, x, m in sorted(detections, key=lambda d: -d[0]):
            region = labels[y : y + m.shape[0], x : x + m.shape[1]]
            free = m & (region == 0)
            if free.sum() >= 0.5 * m.sum():
                region[free] = next_id
                next_id += 1
        return Result(str(path), next_id - 1, labels, img)


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
    ap = argparse.ArgumentParser(description="Count items with the fine-tuned YOLO11-seg model.")
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("-o", "--out", default="outputs", help="folder for overlays and counts.json")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counter = Counter(args.weights, args.conf)
    counts = {}
    for f in image_files(args.inputs):
        res = counter(f)
        cv2.imwrite(str(out / f"{f.parent.name}__{f.stem}.jpg"), overlay(res), [cv2.IMWRITE_JPEG_QUALITY, 90])
        counts[f"{f.parent.name}/{f.name}"] = res.count
        print(f"{f.parent.name}/{f.name:28s} {res.count}", flush=True)
    (out / "counts.json").write_text(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
