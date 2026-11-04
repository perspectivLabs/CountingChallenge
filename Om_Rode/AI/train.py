"""Train YOLOv8s-seg on the SAM auto-labelled, tiled dataset.

Run prepare_dataset.py first. Training uses 1024 px tiles; heavy geometric/colour
augmentation compensates for the small number of source images.
"""
import argparse
import os

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Om_Rode/AI/dataset/data.yaml")
    ap.add_argument("--weights", default="weights/yolov8s-seg.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    model = YOLO(args.weights)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        project=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"), name="yolov8s_seg", exist_ok=True,
        patience=40, max_det=1000, overlap_mask=True,
        degrees=180, flipud=0.5, fliplr=0.5, scale=0.4, translate=0.1,
        hsv_h=0.05, hsv_s=0.5, hsv_v=0.4, mosaic=0.0, close_mosaic=0,
        workers=4, seed=0, plots=True,
    )


if __name__ == "__main__":
    main()
