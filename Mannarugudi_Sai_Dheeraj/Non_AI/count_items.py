"""Count dark, separated objects on a light background with OpenCV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _watershed_split(mask: np.ndarray, distance_threshold: float = 0.45) -> int:
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if distance.max() == 0:
        return 1

    _, sure_foreground = cv2.threshold(
        distance, distance_threshold * distance.max(), 255, cv2.THRESH_BINARY
    )
    sure_foreground = np.uint8(sure_foreground)
    sure_background = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_background, sure_foreground)
    _, markers = cv2.connectedComponents(sure_foreground)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
    return len(np.unique(markers[markers > 1]))


def _estimate_group_size(area: int, single_object_area: int, mask: np.ndarray) -> int:
    ratio = area / single_object_area
    if ratio < 1.6:
        return 1
    if ratio < 4.0:
        return max(2, round(ratio))
    return max(_watershed_split(mask), max(2, round(ratio * 0.9)))


def _add_border(image: np.ndarray, size: int = 30) -> np.ndarray:
    background = image[min(10, image.shape[0] - 1), min(10, image.shape[1] - 1)].tolist()
    return cv2.copyMakeBorder(
        image, size, size, size, size, cv2.BORDER_CONSTANT, value=background
    )


def segment(
    image: np.ndarray,
    min_area: int | None = None,
    strategy: str = "area",
    single_object_area: int = 3506,
) -> tuple[int, np.ndarray, list[np.ndarray]]:
    """Return estimated count, binary foreground mask, and component contours."""
    if strategy not in {"components", "area", "nuts"}:
        raise ValueError(f"Unknown counting strategy: {strategy}")
    if single_object_area <= 0:
        raise ValueError("single_object_area must be greater than zero")

    if strategy == "components":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        _, foreground = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        height, width = gray.shape
        scale = max(1, min(height, width) // 500)
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * scale + 1, 2 * scale + 1)
        )
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, close_kernel)
    elif strategy == "area":
        working_image = _add_border(image)
        gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, foreground = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        kernel = np.ones((3, 3), np.uint8)
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, kernel, iterations=1
        )
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_CLOSE, kernel, iterations=2
        )
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41,
            2,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, kernel, iterations=1
        )

    height, width = foreground.shape
    default_area_floor = (
        max(30, int(height * width * 0.00002))
        if strategy == "components"
        else 100 if strategy == "nuts" else 500
    )
    area_floor = min_area if min_area is not None else default_area_floor
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)

    single_nut_area = None
    if strategy == "nuts":
        nut_areas = [
            stats[label, cv2.CC_STAT_AREA]
            for label in range(1, count)
            if 200 < stats[label, cv2.CC_STAT_AREA] < 5000
        ]
        if nut_areas:
            single_nut_area = float(np.median(nut_areas))

    cleaned = np.zeros_like(foreground)
    estimated_count = 0
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < area_floor:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        component_width = stats[label, cv2.CC_STAT_WIDTH]
        component_height = stats[label, cv2.CC_STAT_HEIGHT]
        if strategy in {"area", "nuts"} and (
            x < 5
            or y < 5
            or x + component_width > width - 5
            or y + component_height > height - 5
        ):
            continue

        component = labels == label
        cleaned[component] = 255
        if strategy == "components":
            estimated_count += 1
        elif strategy == "area":
            component_mask = (
                component[y:y + component_height, x:x + component_width] * 255
            ).astype(np.uint8)
            estimated_count += _estimate_group_size(
                area, single_object_area, component_mask
            )
        elif single_nut_area is not None:
            estimated_count += (
                1
                if area < 1.5 * single_nut_area
                else int(round(area / single_nut_area))
            )

    if strategy == "area":
        cleaned = cleaned[30:-30, 30:-30]
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return estimated_count, cleaned, contours


def render(image: np.ndarray, mask: np.ndarray, contours: list[np.ndarray], count: int) -> np.ndarray:
    overlay = image.copy()
    tint = np.zeros_like(image)
    tint[:, :] = (30, 210, 40)
    selected = mask > 0
    overlay[selected] = cv2.addWeighted(image, 0.45, tint, 0.55, 0)[selected]
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    cv2.putText(
        overlay,
        f"Count: {count}",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    return overlay


def image_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    raise FileNotFoundError(f"Input path does not exist: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Image file or folder of images")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "output",
        help="Folder for overlay images and binary masks",
    )
    parser.add_argument("--min-area", type=int, default=None, help="Minimum foreground component area")
    parser.add_argument(
        "--strategy",
        choices=("components", "area", "nuts"),
        default="area",
        help="Counting method: components, screw area/watershed, or adaptive-threshold nuts",
    )
    parser.add_argument(
        "--single-object-area",
        type=int,
        default=3506,
        help="Expected foreground pixel area of one object for --strategy area",
    )
    args = parser.parse_args()

    files = image_files(args.input)
    if not files:
        raise FileNotFoundError(f"No supported images found under: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "counts.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("image", "predicted_count"))
        writer.writeheader()

        for path in files:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not decode image: {path}")
            count, mask, contours = segment(
                image,
                args.min_area,
                strategy=args.strategy,
                single_object_area=args.single_object_area,
            )
            overlay = render(image, mask, contours, count)
            if not cv2.imwrite(str(args.output / f"{path.stem}_overlay.jpg"), overlay):
                raise OSError(f"Could not write overlay for: {path}")
            if not cv2.imwrite(str(args.output / f"{path.stem}_mask.png"), mask):
                raise OSError(f"Could not write mask for: {path}")
            writer.writerow({"image": str(path), "predicted_count": count})
            print(f"{path.name}: {count}")


if __name__ == "__main__":
    main()
