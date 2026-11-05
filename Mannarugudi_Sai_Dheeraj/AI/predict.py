"""Run a trained YOLO segmentation model and save mask overlays."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_calibration_data(csv_path: Path) -> list[tuple[Path, int]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Calibration CSV does not exist: {csv_path}")

    calibration_data = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"image", "true_count"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Calibration CSV must include image and true_count columns")
        for row_number, row in enumerate(reader, start=2):
            try:
                true_count = int(row["true_count"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid true_count on calibration CSV row {row_number}") from error
            if true_count < 0:
                raise ValueError(f"true_count cannot be negative on calibration CSV row {row_number}")
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = csv_path.resolve().parent / image_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Calibration image on CSV row {row_number} does not exist: {image_path}"
                )
            calibration_data.append((image_path, true_count))

    if not calibration_data:
        raise ValueError("Calibration CSV has no data rows")
    return calibration_data


def calibrate_confidence(
    model: YOLO, calibration_data: list[tuple[Path, int]]
) -> tuple[float, float, float]:
    observations = []
    candidate_thresholds = {0.001, 1.0}
    for image_path, true_count in calibration_data:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not decode calibration image: {image_path}")
        result = model.predict(image, conf=0.001, verbose=False)[0]
        scores = (
            result.boxes.conf.cpu().numpy().tolist()
            if result.boxes is not None else []
        )
        observations.append((scores, true_count))
        candidate_thresholds.update(scores)

    def metrics(threshold: float) -> tuple[float, float]:
        errors = [
            abs(sum(score >= threshold for score in scores) - true_count)
            for scores, true_count in observations
        ]
        mean_absolute_error = sum(errors) / len(errors)
        exact_accuracy = sum(error == 0 for error in errors) / len(errors)
        return mean_absolute_error, exact_accuracy

    threshold = min(
        candidate_thresholds,
        key=lambda threshold: (
            metrics(threshold)[0],
            -metrics(threshold)[1],
            -threshold,
        ),
    )
    mean_absolute_error, exact_accuracy = metrics(threshold)
    return threshold, mean_absolute_error, exact_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="Path to best.pt")
    parser.add_argument("--input", required=True, type=Path, help="Image file or folder")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "predictions"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.02,
        help="Minimum detection confidence; validation calibration is recommended",
    )
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        help="Optional validation CSV with image and true_count columns; overrides --confidence",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {args.weights}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input}")
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    files = (
        [args.input] if args.input.is_file()
        else sorted(p for p in args.input.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    )
    if not files:
        raise FileNotFoundError(f"No supported images found under: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    confidence = args.confidence
    if args.calibration_csv is not None:
        calibration_data = read_calibration_data(args.calibration_csv)
        confidence, validation_mae, validation_accuracy = calibrate_confidence(
            model, calibration_data
        )
        print(
            f"Calibrated confidence: {confidence:.4f} "
            f"(validation MAE: {validation_mae:.3f}, "
            f"exact-count accuracy: {validation_accuracy:.2%}, "
            f"{len(calibration_data)} images)"
        )

    with (args.output / "counts.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("image", "predicted_count"))
        writer.writeheader()
        for path in files:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not decode image: {path}")
            result = model.predict(
                image, conf=0.001 if args.calibration_csv is not None else confidence,
                verbose=False,
            )[0]
            overlay = image.copy()
            boxes = result.boxes
            if boxes is None:
                selected_boxes = []
            else:
                keep = (
                    boxes.conf >= confidence
                    if args.calibration_csv is not None
                    else None
                )
                coordinates = boxes.xyxy.cpu().numpy()
                if keep is not None:
                    coordinates = coordinates[keep.cpu().numpy()]
                selected_boxes = coordinates.astype(int)

            count = len(selected_boxes)
            if count:
                box_overlay = overlay.copy()
                for x1, y1, x2, y2 in selected_boxes:
                    cv2.rectangle(
                        box_overlay, (x1, y1), (x2, y2), (0, 255, 255), -1
                    )
                overlay = cv2.addWeighted(box_overlay, 0.25, overlay, 0.75, 0)
                for x1, y1, x2, y2 in selected_boxes:
                    cv2.rectangle(
                        overlay, (x1, y1), (x2, y2), (0, 180, 255), 2
                    )
            cv2.putText(
                overlay, f"Count: {count}", (24, 48), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3, cv2.LINE_AA,
            )
            destination = args.output / f"{path.stem}_prediction.jpg"
            if not cv2.imwrite(str(destination), overlay):
                raise OSError(f"Could not write prediction: {destination}")
            writer.writerow({"image": str(path), "predicted_count": count})
            print(f"{path.name}: {count}")


if __name__ == "__main__":
    main()
