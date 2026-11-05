"""
AI Item Counting and Mask Overlay Pipeline
Author: Athira PT
Task: Count screws/bolts in images and overlay instance masks using Deep Learning / Foundation Models.
Accuracy Target: > 95%
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np
import torch
from ultralytics import FastSAM, YOLO


def get_ai_model(model_type="fastsam", weights_path="Athira_PT/AI/models/best.pt"):
    """
    Loads either the FastSAM foundation model or custom fine-tuned YOLOv8-seg model.
    """
    device = 0 if torch.cuda.is_available() else "cpu"
    if model_type == "fastsam":
        model_name = "FastSAM-s.pt"
        model = FastSAM(model_name)
    else:
        if os.path.exists(weights_path):
            model = YOLO(weights_path)
        else:
            print(f"Custom weights not found at {weights_path}, defaulting to FastSAM-s.pt")
            model = FastSAM("FastSAM-s.pt")
            model_type = "fastsam"
    return model, device, model_type


def detect_items_ai(
    image_input,
    model=None,
    model_type="fastsam",
    conf=0.55,
    iou=0.85,
    min_mask_area=100,
    overlay_alpha=0.45,
    imgsz=1024
):
    """
    Detects, segments, and counts items using Deep Learning / Foundation AI models.

    Args:
        image_input: File path (str) or BGR image (np.ndarray).
        model: Pre-instantiated FastSAM or YOLO model (optional).
        model_type: 'fastsam' (foundation model) or 'custom' (YOLOv8-seg).
        conf: Confidence threshold for predictions (default: 0.55).
        iou: IoU threshold for Non-Maximum Suppression (default: 0.85).
        min_mask_area: Minimum pixel area to discard tiny spurious detections (default: 100).
        overlay_alpha: Opacity factor for translucent mask blending (default: 0.45).
        imgsz: Inference resolution (default: 1024).

    Returns:
        dict containing:
            - 'count': Total detected item count
            - 'items': List of item metadata (id, bbox, centroid, mask, conf)
            - 'annotated_image': Visual overlay with translucent masks, boxes, labels, and banner
            - 'overlay_image': Clean image overlay without banner
            - 'binary_mask': Composite binary segmentation mask
            - 'original_image': Unmodified input image
    """
    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image not found at: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not read image file: {image_input}")
    else:
        img = image_input.copy()

    orig_h, orig_w = img.shape[:2]

    # Load model if not provided
    if model is None:
        model, device, model_type = get_ai_model(model_type)
    else:
        device = 0 if torch.cuda.is_available() else "cpu"

    # Run inference
    with torch.no_grad():
        if model_type == "fastsam":
            results = model(
                img,
                device=device,
                retina_masks=True,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False
            )
        else:
            results = model.predict(
                img,
                device=device,
                retina_masks=True,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False
            )

    res = results[0]

    # Parse masks
    composite_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    mask_layer = np.zeros_like(img)
    overlay = img.copy()

    items = []
    item_counter = 0

    color_palette = [
        (0, 255, 0),     # Bright Green
        (255, 0, 0),     # Blue
        (0, 165, 255),   # Orange
        (255, 255, 0),   # Cyan
        (0, 255, 255),   # Yellow
        (255, 0, 255),   # Magenta
        (128, 0, 255),   # Purple
        (0, 215, 255),   # Gold
        (50, 205, 50),   # Lime
        (255, 144, 30)   # Dodger Blue
    ]

    scale_factor = max(orig_h, orig_w) / 1080.0

    if res.masks is not None and len(res.masks) > 0:
        masks_tensor = res.masks.data
        for idx in range(len(masks_tensor)):
            m_np = masks_tensor[idx].cpu().numpy().astype(np.uint8)
            
            # Resize mask to original image dimensions if needed
            if m_np.shape[:2] != (orig_h, orig_w):
                m_np = cv2.resize(m_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            area = np.sum(m_np > 0)
            if area < min_mask_area:
                continue

            item_counter += 1
            color = color_palette[(item_counter - 1) % len(color_palette)]

            # Add to composite binary mask
            composite_mask[m_np > 0] = 255

            # Translucent mask layer
            mask_layer[m_np > 0] = color

            # Contours and bounding box
            cnts, _ = cv2.findContours(m_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            largest_cnt = max(cnts, key=cv2.contourArea)
            
            # Outline
            cv2.drawContours(overlay, [largest_cnt], -1, color, thickness=max(2, int(round(2 * scale_factor))))

            x, y, w, h = cv2.boundingRect(largest_cnt)
            M = cv2.moments(largest_cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2

            # Centroid point
            cv2.circle(overlay, (cx, cy), max(3, int(round(4 * scale_factor))), (0, 0, 255), -1)

            # Bounding box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (200, 200, 200), 1)

            # Label
            font_scale = max(0.4, 0.55 * scale_factor)
            thickness = max(1, int(round(1.5 * scale_factor)))
            cv2.putText(overlay, f"#{item_counter}", (x, max(18, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

            items.append({
                "id": item_counter,
                "bbox": [int(x), int(y), int(w), int(h)],
                "centroid": (int(cx), int(cy)),
                "area": int(area)
            })

    # Alpha blend overlay
    cv2.addWeighted(mask_layer, overlay_alpha, overlay, 1.0 - overlay_alpha, 0, overlay)

    # Top banner
    banner_height = max(55, int(round(65 * scale_factor)))
    banner = np.zeros((banner_height, orig_w, 3), dtype=np.uint8)
    banner[:] = (25, 25, 25)
    
    header_text = f"AI Detected Count: {len(items)} items  |  Model: {model_type.upper()} Instance Segmentation"
    font_scale_hdr = max(0.6, 0.8 * scale_factor)
    cv2.putText(banner, header_text, (20, int(banner_height * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_hdr, (0, 255, 0), max(1, int(round(2 * scale_factor))), cv2.LINE_AA)

    final_annotated = np.vstack([banner, overlay])

    return {
        "count": len(items),
        "items": items,
        "annotated_image": final_annotated,
        "overlay_image": overlay,
        "binary_mask": composite_mask,
        "original_image": img
    }


def evaluate_dataset_ai(dataset_dir, output_dir=None, model_type="fastsam", conf=0.55):
    """
    Evaluates AI detection across all images in a folder and saves outputs.
    """
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext.upper())))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No images found in {dataset_dir}")
        return []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"AI EVALUATION ON DATASET: {os.path.basename(dataset_dir)} ({len(image_paths)} images)")
    print(f"Model: {model_type.upper()}  |  Confidence: {conf}")
    print(f"{'='*70}")
    print(f"{'Filename':<35} | {'Detected Count':<16} | {'Status':<10}")
    print(f"{'-'*70}")

    model, device, m_type = get_ai_model(model_type)

    results = []
    for img_p in image_paths:
        fname = os.path.basename(img_p)
        res = detect_items_ai(img_p, model=model, model_type=m_type, conf=conf)
        count = res["count"]
        results.append({"filename": fname, "count": count, "path": img_p})
        print(f"{fname:<35} | {count:<16} | {'OK':<10}")

        if output_dir:
            out_img_name = f"ai_detected_{os.path.splitext(fname)[0]}.jpg"
            out_mask_name = f"ai_mask_{os.path.splitext(fname)[0]}.png"
            cv2.imwrite(os.path.join(output_dir, out_img_name), res["annotated_image"])
            cv2.imwrite(os.path.join(output_dir, out_mask_name), res["binary_mask"])

    print(f"{'-'*70}")
    counts = [r["count"] for r in results]
    print(f"Total Images: {len(results)} | Mean Count: {np.mean(counts):.1f} | Min: {min(counts)} | Max: {max(counts)}")
    print(f"{'='*70}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="AI Item Counting and Mask Overlay using Deep Learning")
    parser.add_argument("--input", type=str, default="data/Screws", help="Path to image file or directory")
    parser.add_argument("--output_dir", type=str, default="Athira_PT/AI/output", help="Directory to save output visualizations")
    parser.add_argument("--model", type=str, default="fastsam", choices=["fastsam", "custom"], help="AI model to use")
    parser.add_argument("--conf", type=float, default=0.55, help="Confidence threshold")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        evaluate_dataset_ai(args.input, output_dir=args.output_dir, model_type=args.model, conf=args.conf)
    elif os.path.isfile(args.input):
        res = detect_items_ai(args.input, model_type=args.model, conf=args.conf)
        print(f"\nProcessed {args.input}: Detected {res['count']} items.")
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            fname = os.path.basename(args.input)
            out_img = os.path.join(args.output_dir, f"ai_detected_{fname}")
            out_mask = os.path.join(args.output_dir, f"ai_mask_{os.path.splitext(fname)[0]}.png")
            cv2.imwrite(out_img, res["annotated_image"])
            cv2.imwrite(out_mask, res["binary_mask"])
            print(f"Saved annotated overlay to: {out_img}")
            print(f"Saved binary mask to: {out_mask}")
    else:
        print(f"Error: Input path '{args.input}' does not exist.")


if __name__ == "__main__":
    main()
