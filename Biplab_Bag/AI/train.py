"""Fine-tune YOLO11-seg (COCO-pretrained) on the synthetic copy-paste dataset.

    python train.py --data dataset/data.yaml --epochs 50 --imgsz 512 --batch 4

The best checkpoint is copied to  weights/best.pt  (used by count_ai.py).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "dataset" / "data.yaml"))
    ap.add_argument("--model", default="yolo11s-seg.pt", help="pretrained starting point")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=512)  # 1024 px tiles are resized to this
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(HERE / "runs"),
        name="items_seg",
        exist_ok=True,
        seed=0,
        deterministic=False,
        cos_lr=True,
        patience=15,
        close_mosaic=10,
        # geometry: items are already rotated / scaled during synthesis, so only
        # mild extra augmentation; count_ai.py runs inference at the same scale
        degrees=10.0,
        scale=0.2,
        translate=0.1,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        mixup=0.0,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.3,
        max_det=1000,          # dense piles hold > 300 items
        plots=True,
        amp=True,
    )
    best = HERE / "runs" / "items_seg" / "weights" / "best.pt"
    (HERE / "weights").mkdir(exist_ok=True)
    shutil.copy(best, HERE / "weights" / "best.pt")
    print(f"best weights -> {HERE / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
