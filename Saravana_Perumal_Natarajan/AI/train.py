"""Fine-tune COCO-pretrained YOLO11n-seg on the tiled dataset from ``make_dataset.py``.

Writes ``runs/segment/items/weights/best.pt`` (copy it to ``model/`` to use it as the default).

Usage:
    uv run train.py
"""

from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def device() -> str:
    """Pick Apple MPS, CUDA or CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    """Train with the settings used for the submitted model."""
    model = YOLO(str(HERE / "weights" / "yolo11n-seg.pt"))  # downloaded on first use
    model.train(
        data=str(HERE / "datasets" / "items" / "data.yaml"),
        imgsz=640,
        epochs=120,
        patience=40,
        batch=8,
        device=device(),
        project=str(HERE / "runs" / "segment"),
        name="items",
        exist_ok=True,
        max_det=600,  # a 640 px tile can hold over 100 nuts
        degrees=180,  # objects lie at any angle
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=15,
        seed=0,
    )


if __name__ == "__main__":
    main()
