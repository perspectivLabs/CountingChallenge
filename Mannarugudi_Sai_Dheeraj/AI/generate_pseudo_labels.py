"""Export provisional YOLO segmentation labels from the OpenCV foreground mask.

These are pseudo-labels for review, not human-verified ground truth.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent / "Non_AI"))
from count_items import IMAGE_SUFFIXES, segment  # noqa: E402


def group_key(path: Path, source: Path) -> str:
    relative = path.relative_to(source)
    stem = path.stem.lower()
    if stem.startswith("img1"):
        stem = "img1"
    return f"{relative.parent.as_posix()}::{stem}"


def polygon_lines(mask, width: int, height: int) -> list[str]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.003), True)
        points = polygon.reshape(-1, 2)
        if len(points) < 3:
            continue
        normalized = []
        for x, y in points:
            normalized.extend((f"{x / width:.6f}", f"{y / height:.6f}"))
        lines.append("0 " + " ".join(normalized))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "Mannarugudi_Sai_Dheeraj" / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "annotated_hardware",
        help="Output dataset folder (ignored by Git)",
    )
    parser.add_argument("--min-area", type=int, default=None)
    args = parser.parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"Image dataset directory does not exist: {args.input}")

    images = sorted(
        path for path in args.input.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {args.input}")

    groups = sorted({group_key(path, args.input) for path in images})
    validation_groups = set(groups[::5]) if len(groups) >= 5 else set()
    if not validation_groups:
        raise ValueError("At least five distinct image groups are required for a train/validation split")
    if len(validation_groups) == len(groups):
        validation_groups.remove(groups[-1])

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": [0, 0], "val": [0, 0]}
    for image_path in images:
        split = "val" if group_key(image_path, args.input) in validation_groups else "train"
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not decode image: {image_path}")
        count, mask, _ = segment(image, args.min_area)
        lines = polygon_lines(mask, image.shape[1], image.shape[0])

        # Prefix filenames to avoid collisions between source folders.
        relative = image_path.relative_to(args.input)
        safe_name = f"{relative.parent.name}_{image_path.stem}"
        image_destination = args.output / "images" / split / f"{safe_name}{image_path.suffix.lower()}"
        label_destination = args.output / "labels" / split / f"{safe_name}.txt"
        shutil.copy2(image_path, image_destination)
        label_destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats[split][0] += 1
        stats[split][1] += len(lines)
        print(f"{split}: {image_path.name}: {len(lines)} provisional instances (connected components: {count})")

    yaml_path = SCRIPT_DIR / "hardware.yaml"
    yaml_path.write_text(
        "path: annotated_hardware\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: hardware\n",
        encoding="utf-8",
    )
    print(f"Dataset YAML: {yaml_path}")
    print(f"Images copied to: {args.output}")
    print(f"Train: {stats['train'][0]} images, {stats['train'][1]} provisional instances")
    print(f"Validation: {stats['val'][0]} images, {stats['val'][1]} provisional instances")
    print("WARNING: labels are OpenCV pseudo-labels; manually review and correct them before training.")


if __name__ == "__main__":
    main()
