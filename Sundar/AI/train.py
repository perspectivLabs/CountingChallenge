"""
Fine-tune YOLOv8 segmentation model on custom counting dataset.

Training Method:
- Base model: YOLOv8n-seg (pretrained on COCO)
- Fine-tuned on the provided counting dataset
- Uses instance segmentation for precise mask overlay

Steps:
1. Organize dataset in YOLO format (images/ and labels/ folders)
2. Create a data.yaml pointing to train/val splits
3. Run this script to fine-tune

Dataset format (YOLO segmentation):
  dataset/
    images/
      train/
      val/
    labels/
      train/
      val/
    data.yaml
"""
from ultralytics import YOLO
import argparse


def train(data_yaml, epochs=50, imgsz=640, model_path="yolov8n-seg.pt"):
    model = YOLO(model_path)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name="counting_model",
        patience=10,
    )
    print("Training complete. Best model saved in runs/segment/counting_model/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg for counting")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt")
    args = parser.parse_args()

    train(args.data, args.epochs, args.imgsz, args.model)
