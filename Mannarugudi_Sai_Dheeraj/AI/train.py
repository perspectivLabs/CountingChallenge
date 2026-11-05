"""Fine-tune a pretrained YOLO instance-segmentation model on annotated hardware."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from ultralytics import YOLO
import yaml


def resolved_dataset_yaml(data_yaml: Path, destination: Path) -> Path:
    with data_yaml.open(encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_yaml}")

    dataset_root = Path(config.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.resolve().parent / dataset_root
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}")

    config["path"] = str(dataset_root)
    with destination.open("w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(config, yaml_file, sort_keys=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", required=True, type=Path,
        help="Ultralytics dataset YAML exported with instance-segmentation polygon labels",
    )
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Pretrained segmentation weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="For example: 0, cpu, or mps")
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {args.data}")
    if args.epochs < 1 or args.imgsz < 1 or args.batch < 1:
        parser.error("--epochs, --imgsz, and --batch must be positive")

    with tempfile.TemporaryDirectory(prefix="hardware-dataset-") as temp_dir:
        runtime_yaml = resolved_dataset_yaml(
            args.data, Path(temp_dir) / args.data.name
        )
        model = YOLO(args.model)
        model.train(
            data=str(runtime_yaml),
            task="segment",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=20,
            project=str(Path(__file__).parent / "runs"),
            name="hardware-segmentation",
        )


if __name__ == "__main__":
    main()
