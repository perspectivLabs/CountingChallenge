from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

AI_DIR = Path(__file__).resolve().parent


def train_yolo_medium_seg(
    model_name: str,
    data_yaml: Path,
    epochs: int,
    patience: int,
    imgsz: int,
    batch: int,
    device: str | int,
    workers: int,
    project_dir: Path,
    run_name: str,
    out_weights: Path,
    weights_dir: Path,
) -> None:
    data_yaml = data_yaml.resolve()
    project_dir = project_dir.resolve()
    out_weights = out_weights.resolve()
    weights_dir = weights_dir.resolve()

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}. Run generate_synthetic_dataset.py first."
        )

    weights_dir.mkdir(parents=True, exist_ok=True)
    out_weights.parent.mkdir(parents=True, exist_ok=True)

    base_weights = weights_dir / model_name
    prev_cwd = Path.cwd()
    try:
        # Keep pretrained base downloads and any Ultralytics AMP check checkpoints inside AI/weights/
        os.chdir(weights_dir)
        model = YOLO(str(base_weights))
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            overlap_mask=False,
            mask_ratio=2,
            max_det=300,
            degrees=180.0,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.5,
            close_mosaic=min(10, max(1, epochs // 5)),
            scale=0.30,
            hsv_v=0.25,
            hsv_s=0.25,
            plots=True,
            seed=42,
        )
    finally:
        os.chdir(prev_cwd)

    best_pt = project_dir / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Expected best checkpoint missing at: {best_pt}")

    shutil.copy2(best_pt, out_weights)
    print(f"\nSaved trained medium YOLO-seg weights to: {out_weights}")

    if results is not None:
        box_map50 = getattr(getattr(results, "box", None), "map50", None)
        box_map = getattr(getattr(results, "box", None), "map", None)
        seg_map50 = getattr(getattr(results, "seg", None), "map50", None)
        seg_map = getattr(getattr(results, "seg", None), "map", None)
        print(
            f"Validation metrics -> box.map50={box_map50:.4f}, box.map={box_map:.4f}, "
            f"seg.map50={seg_map50:.4f}, seg.map={seg_map:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train medium YOLO11m-seg on the synthetic modal-occlusion dataset"
    )
    parser.add_argument("--model", type=str, default="yolo11m-seg.pt")
    parser.add_argument("--data", type=Path, default=AI_DIR / "synthetic_dataset" / "data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=AI_DIR / "runs")
    parser.add_argument("--name", type=str, default="yolo11m_synthetic_seg")
    parser.add_argument(
        "--out-weights",
        type=Path,
        default=AI_DIR / "weights" / "yolo11m_synthetic_seg.pt",
    )
    parser.add_argument("--weights-dir", type=Path, default=AI_DIR / "weights")
    args = parser.parse_args()

    device: str | int = int(args.device) if args.device.isdigit() else args.device

    train_yolo_medium_seg(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project_dir=args.project,
        run_name=args.name,
        out_weights=args.out_weights,
        weights_dir=args.weights_dir,
    )


if __name__ == "__main__":
    main()
