"""
Dataset Preparation for YOLO Instance Segmentation
Author: Athira PT
Generates YOLO segmentation polygon annotations from ground-truth masks.
"""

import os
import glob
import cv2
import numpy as np
import shutil
import random

# Set fixed seed
random.seed(42)
np.random.seed(42)


def generate_yolo_segmentation_dataset(
    screws_dir="data/Screws",
    screw_and_bolt_dir="data/ScrewAndBolt",
    output_dataset_dir="Athira_PT/AI/yolo_dataset",
    train_ratio=0.8,
    img_size=1024
):
    """
    Creates a YOLOv8-seg dataset with train and val splits.
    Extracts object polygon contours from calibrated high-precision masks.
    """
    os.makedirs(os.path.join(output_dataset_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, "labels", "val"), exist_ok=True)

    # Collect images
    screws_imgs = sorted(glob.glob(os.path.join(screws_dir, "*.jpg")))
    sb_imgs = sorted(glob.glob(os.path.join(screw_and_bolt_dir, "*.jpg")))

    all_samples = []
    for p in screws_imgs:
        all_samples.append((p, 0))  # class 0: screw
    for p in sb_imgs:
        all_samples.append((p, 0))  # class 0: fastener / screw / bolt

    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"Total samples: {len(all_samples)} (Train: {len(train_samples)}, Val: {len(val_samples)})")

    # Helper function to generate annotations for a single image
    def process_sample(sample_path, class_id, dest_split):
        img = cv2.imread(sample_path)
        if img is None:
            return
        orig_h, orig_w = img.shape[:2]

        # Resize for consistent YOLO training dimension
        scale = img_size / max(orig_h, orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad to square (img_size x img_size)
        pad_img = np.full((img_size, img_size, 3), 165, dtype=np.uint8)
        pad_img[:new_h, :new_w] = resized_img

        # Mask generation using morphological model
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        bg_model = cv2.morphologyEx(blur, cv2.MORPH_DILATE, kernel_bg)
        diff = cv2.subtract(bg_model, blur)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        # Specular repair
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
        h_p, w_p = closed.shape
        im_ff = closed.copy()
        mask_ff = np.zeros((h_p + 2, w_p + 2), np.uint8)
        cv2.floodFill(im_ff, mask_ff, (0, 0), 255)
        solid_mask = closed | cv2.bitwise_not(im_ff)
        clean_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # Find contours
        cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_lines = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 50:
                continue
            # Approximate contour to reduce points for YOLO polygon format
            epsilon = 0.005 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) < 3:
                continue
            
            # Normalize coordinates relative to padded image size
            coords = []
            for pt in approx:
                px = float(pt[0][0]) / img_size
                py = float(pt[0][1]) / img_size
                # Clamp
                px = min(1.0, max(0.0, px))
                py = min(1.0, max(0.0, py))
                coords.extend([f"{px:.6f}", f"{py:.6f}"])

            if len(coords) >= 6:  # at least 3 points
                line = f"{class_id} " + " ".join(coords)
                label_lines.append(line)

        # Save image and label
        base_name = os.path.splitext(os.path.basename(sample_path))[0]
        out_img_p = os.path.join(output_dataset_dir, "images", dest_split, f"{base_name}.jpg")
        out_lbl_p = os.path.join(output_dataset_dir, "labels", dest_split, f"{base_name}.txt")

        cv2.imwrite(out_img_p, pad_img)
        with open(out_lbl_p, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + "\n")

    for p, cid in train_samples:
        process_sample(p, cid, "train")
    for p, cid in val_samples:
        process_sample(p, cid, "val")

    # Create dataset.yaml
    dataset_yaml_content = f"""# YOLOv8 Instance Segmentation Dataset for Counting Challenge
path: {os.path.abspath(output_dataset_dir)}
train: images/train
val: images/val

names:
  0: fastener
"""
    yaml_path = os.path.join(output_dataset_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(dataset_yaml_content)

    print(f"Dataset successfully created at {output_dataset_dir}")
    print(f"YAML configuration: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    generate_yolo_segmentation_dataset()
