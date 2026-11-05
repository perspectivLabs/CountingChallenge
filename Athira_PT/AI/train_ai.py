"""
AI Model Training for Item Instance Segmentation (YOLOv8-seg)
Author: Athira PT
Task: Train Deep Learning model to detect, segment, and count screws/bolts.
"""

import os
import shutil
from ultralytics import YOLO
import torch


def train_yolo_seg(
    data_yaml="Athira_PT/AI/yolo_dataset/dataset.yaml",
    base_model="yolov8n-seg.pt",
    epochs=25,
    imgsz=1024,
    batch=4,
    save_model_dir="Athira_PT/AI/models"
):
    """
    Fine-tunes YOLOv8-seg on the dataset using PyTorch with CUDA acceleration.
    """
    os.makedirs(save_model_dir, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Starting YOLOv8-seg training on device: {device} ({torch.cuda.get_device_name(0) if device==0 else 'CPU'})")
    print(f"Dataset config: {os.path.abspath(data_yaml)}")

    # Initialize model
    model = YOLO(base_model)

    # Train
    results = model.train(
        data=os.path.abspath(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="Athira_PT/AI/runs",
        name="segmentation_exp",
        exist_ok=True,
        plots=True,
        verbose=True
    )

    # Best weights path
    best_weights = os.path.join("Athira_PT/AI/runs", "segmentation_exp", "weights", "best.pt")
    target_weights = os.path.join(save_model_dir, "best.pt")

    if os.path.exists(best_weights):
        shutil.copy(best_weights, target_weights)
        print(f"\nSuccessfully trained! Best model weights saved to: {target_weights}")
    else:
        print(f"\nWarning: {best_weights} not found. Saving last available model.")
        model.save(target_weights)

    # Run validation
    print("\nRunning validation on validation set...")
    val_model = YOLO(target_weights)
    metrics = val_model.val(data=os.path.abspath(data_yaml), imgsz=imgsz, device=device)
    
    print("\n" + "="*60)
    print("VALIDATION METRICS SUMMARY:")
    print("="*60)
    print(f"Mask mAP@0.5:      {metrics.seg.map50:.4f}")
    print(f"Mask mAP@0.5:0.95: {metrics.seg.map:.4f}")
    print(f"Box mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"Box mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print("="*60)

    return target_weights, metrics


if __name__ == "__main__":
    train_yolo_seg()
