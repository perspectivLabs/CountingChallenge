from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VAL_STEMS = {
    "20240713_193135",
    "20240713_193659",
    "20240713_193907",
    "20240713_194232",
    "20240713_194630",
    "img4",
}
CLASS_MAP = {"screws": 0, "nuts": 1}


def generate_tiles(
    h: int,
    w: int,
    tile_size: int = 1024,
    overlap: int = 256,
) -> list[tuple[int, int, int, int]]:
    """Generate sliding-window tile bounds (tx0, ty0, tx1, ty1) covering the full image."""
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_size must be larger than overlap")

    def _starts(dim: int) -> list[int]:
        if dim <= tile_size:
            return [0]
        starts = list(range(0, dim - tile_size + 1, stride))
        last = dim - tile_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    xs = _starts(w)
    ys = _starts(h)
    return [(tx, ty, min(w, tx + tile_size), min(h, ty + tile_size)) for ty in ys for tx in xs]


def load_scene_classes(summary_csv: Path) -> dict[str, int]:
    """Map image filename -> class id (0=screw, 1=nut) from Non_AI summary CSV."""
    mapping: dict[str, int] = {}
    with summary_csv.open("r", newline="") as f:
        for row in csv.DictReader(f):
            scene = row["scene_type"].strip().lower()
            if scene in CLASS_MAP:
                mapping[row["image"].strip()] = CLASS_MAP[scene]
    return mapping


def extract_tile_polygons(
    tile_mask: np.ndarray,
    full_areas: np.ndarray,
    cls_id: int,
    min_tile_area: int = 60,
    min_keep_ratio: float = 0.40,
) -> list[str]:
    """Convert non-background uint16 instances inside a tile to YOLO-seg polygon lines."""
    th, tw = tile_mask.shape
    counts = np.bincount(tile_mask.ravel())
    labels = np.nonzero(counts)[0]

    lines: list[str] = []
    for lbl in labels:
        if lbl == 0:
            continue
        tile_area = int(counts[lbl])
        full_area = int(full_areas[lbl])
        if tile_area < min_tile_area or full_area <= 0:
            continue
        if tile_area < min_keep_ratio * full_area:
            continue

        inst_bin = (tile_mask == lbl).astype(np.uint8)
        cnts, _ = cv2.findContours(inst_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        perim = cv2.arcLength(cnt, True)
        if perim <= 1e-6:
            continue
        approx = cv2.approxPolyDP(cnt, max(1.0, 0.004 * perim), True)
        if len(approx) < 3:
            approx = cnt
        if len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0] / float(tw), 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / float(th), 0.0, 1.0)

        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append(f"{cls_id} {coords}")

    return lines


def prepare_dataset(
    images_dir: Path,
    label_masks_dir: Path,
    summary_csv: Path,
    dataset_dir: Path,
    tile_size: int = 1024,
    overlap: int = 256,
) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    scene_classes = load_scene_classes(summary_csv)
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    stats = {
        "train": {"tiles": 0, "empty_tiles": 0, "cls_0": 0, "cls_1": 0},
        "val": {"tiles": 0, "empty_tiles": 0, "cls_0": 0, "cls_1": 0},
    }
    empty_counter = 0

    for img_path in image_paths:
        stem = img_path.stem
        split = "val" if stem in VAL_STEMS else "train"
        if img_path.name not in scene_classes:
            raise KeyError(f"Missing scene_type for {img_path.name} in {summary_csv}")
        cls_id = scene_classes[img_path.name]

        mask_path = label_masks_dir / f"{stem}_labels.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing label mask: {mask_path}")

        img = cv2.imread(str(img_path))
        label_map = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None or label_map is None:
            raise RuntimeError(f"Failed reading {img_path} or {mask_path}")
        if label_map.ndim == 3:
            label_map = label_map[:, :, 0]

        h, w = img.shape[:2]
        full_areas = np.bincount(label_map.ravel())
        tiles = generate_tiles(h, w, tile_size=tile_size, overlap=overlap)

        for tx0, ty0, tx1, ty1 in tiles:
            tile_img = img[ty0:ty1, tx0:tx1]
            tile_mask = label_map[ty0:ty1, tx0:tx1]
            poly_lines = extract_tile_polygons(tile_mask, full_areas, cls_id)

            if not poly_lines:
                empty_counter += 1
                # Keep ~10% of pure-empty tiles as negative background examples
                if empty_counter % 10 != 0:
                    continue
                stats[split]["empty_tiles"] += 1

            tile_name = f"{stem}_r{ty0}_c{tx0}"
            out_img = dataset_dir / "images" / split / f"{tile_name}.jpg"
            out_lbl = dataset_dir / "labels" / split / f"{tile_name}.txt"

            cv2.imwrite(str(out_img), tile_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out_lbl.write_text("\n".join(poly_lines) + ("\n" if poly_lines else ""))

            stats[split]["tiles"] += 1
            stats[split][f"cls_{cls_id}"] += len(poly_lines)

    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "screw", 1: "nut"},
    }
    yaml_path = dataset_dir / "data.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"Prepared tiled YOLO-seg dataset at: {dataset_dir}")
    print(f"Written data.yaml: {yaml_path}")
    for split in ("train", "val"):
        s = stats[split]
        print(
            f"  {split:5s}: tiles={s['tiles']:4d} (empty={s['empty_tiles']:2d}) | "
            f"screw_insts={s['cls_0']:4d}, nut_insts={s['cls_1']:4d}"
        )


if __name__ == "__main__":
    ai_dir = Path(__file__).resolve().parent
    root_dir = ai_dir.parent
    prepare_dataset(
        images_dir=root_dir / "images",
        label_masks_dir=root_dir / "Non_AI" / "adaptive_results" / "06_final_adaptive" / "label_masks",
        summary_csv=root_dir / "Non_AI" / "adaptive_results" / "summary_counts.csv",
        dataset_dir=ai_dir / "dataset",
    )
