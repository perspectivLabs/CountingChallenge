"""
AI-based Item Counting using YOLOv8 Instance Segmentation
Uses a pretrained YOLOv8 segmentation model to detect, count, and overlay masks.
"""
import cv2
import numpy as np
import os
import glob
import argparse
from ultralytics import YOLO


def count_and_mask(image_path, model, output_dir="output", conf=0.25):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read: {image_path}")
        return 0

    results = model(image_path, conf=conf, verbose=False)[0]

    count = len(results.boxes) if results.boxes is not None else 0

    # Create mask overlay
    mask_overlay = img.copy()
    np.random.seed(42)

    if results.masks is not None:
        for i, mask in enumerate(results.masks.data):
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
            color = np.random.randint(50, 255, 3).tolist()
            colored_mask = np.zeros_like(img)
            colored_mask[:] = color
            mask_bool = mask_resized > 0.5
            mask_overlay[mask_bool] = (
                0.5 * mask_overlay[mask_bool] + 0.5 * colored_mask[mask_bool]
            ).astype(np.uint8)

    # Draw bounding boxes
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(mask_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        mask_overlay, f"Count: {count}", (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
    )

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"masked_{filename}")
    cv2.imwrite(output_path, mask_overlay)
    print(f"{filename}: Count = {count} | Saved: {output_path}")

    return count


def process_dataset(data_dir, output_dir="output", model_path="yolov8n-seg.pt", conf=0.25):
    model = YOLO(model_path)

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(data_dir, ext)))

    if not image_paths:
        print(f"No images found in {data_dir}")
        return

    total = 0
    for path in sorted(image_paths):
        total += count_and_mask(path, model, output_dir, conf)

    print(f"\nProcessed {len(image_paths)} images. Total items counted: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count items using YOLOv8 (AI)")
    parser.add_argument("--data", type=str, default="data", help="Path to image folder")
    parser.add_argument("--output", type=str, default="output", help="Output folder")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    process_dataset(args.data, args.output, args.model, args.conf)
