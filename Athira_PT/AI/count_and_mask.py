"""
AI Object Counting and Mask Overlay Pipeline
Author: Athira PT
Task: Count objects (screws, bolts, fasteners) and overlay precise instance segmentation masks
      with > 95% accuracy using a Pretrained Vision AI Foundation Model (FastSAM).
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np
import torch
from ultralytics import FastSAM


def get_ai_model(weights="FastSAM-s.pt"):
    """Loads the pretrained FastSAM vision foundation model."""
    device = 0 if torch.cuda.is_available() else "cpu"
    # Search common locations
    candidates = [weights, os.path.join(os.path.dirname(__file__), weights), os.path.abspath(weights)]
    resolved = None
    for c in candidates:
        if os.path.exists(c):
            resolved = c
            break
    if resolved is None:
        resolved = "FastSAM-s.pt"
    model = FastSAM(resolved)
    return model, device


def detect_items_ai(
    image_input,
    model=None,
    conf=0.55,
    iou=0.85,
    min_mask_area=60,
    overlay_alpha=0.45,
    imgsz=1024
):
    """
    Detects, segments, and counts objects using the pretrained AI foundation model.

    Parameters:
        image_input (str or np.ndarray): Filepath or BGR image array.
        model (FastSAM, optional): Loaded FastSAM model instance.
        conf (float): Detection confidence threshold.
        iou (float): Non-maximum suppression IoU threshold.
        min_mask_area (int): Minimum pixel area to discard tiny noise artifacts.
        overlay_alpha (float): Alpha blending transparency (0 to 1).
        imgsz (int): Model inference image resolution.

    Returns:
        dict:
            - 'count': Total object count
            - 'items': List of item metadata (id, bbox, centroid, area)
            - 'annotated_image': Visual overlay with labels, masks, and summary banner
            - 'overlay_image': Clean overlay without banner
            - 'binary_mask': Composite binary mask
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

    if model is None:
        model, device = get_ai_model()
    else:
        device = 0 if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        results = model(
            img,
            device=device,
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False
        )

    res = results[0]

    composite_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    mask_layer = np.zeros_like(img)
    overlay = img.copy()

    items = []
    total_idx = 0
    scale_factor = max(orig_h, orig_w) / 1080.0

    # Color palette for distinct visualization
    colors = [
        (0, 255, 0),    # Bright green
        (0, 215, 255),  # Gold/yellow
        (255, 128, 0),  # Blue
        (0, 165, 255),  # Orange
        (255, 0, 128),  # Violet
        (50, 205, 50),  # Lime
    ]

    if res.masks is not None and len(res.masks) > 0:
        masks_tensor = res.masks.data
        for idx in range(len(masks_tensor)):
            m_np = masks_tensor[idx].cpu().numpy().astype(np.uint8)

            if m_np.shape[:2] != (orig_h, orig_w):
                m_np = cv2.resize(m_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            area = int(np.sum(m_np > 0))
            if area < min_mask_area:
                continue

            total_idx += 1
            color = colors[(total_idx - 1) % len(colors)]

            composite_mask[m_np > 0] = 255
            mask_layer[m_np > 0] = color

            cnts, _ = cv2.findContours(m_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            largest_cnt = max(cnts, key=cv2.contourArea)

            cv2.drawContours(overlay, [largest_cnt], -1, color, thickness=max(2, int(round(2 * scale_factor))))

            x, y, w, h = cv2.boundingRect(largest_cnt)
            M = cv2.moments(largest_cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2

            cv2.rectangle(overlay, (x, y), (x + w, y + h), (220, 220, 220), 1)
            cv2.circle(overlay, (cx, cy), max(3, int(round(3.5 * scale_factor))), (0, 0, 255), -1)

            font_scale = max(0.35, 0.45 * scale_factor)
            thickness_txt = max(1, int(round(1.5 * scale_factor)))
            label = f"#{total_idx}"
            cv2.putText(overlay, label, (x, max(18, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness_txt, cv2.LINE_AA)

            items.append({
                "id": total_idx,
                "bbox": [int(x), int(y), int(w), int(h)],
                "centroid": (int(cx), int(cy)),
                "area": area
            })

    cv2.addWeighted(mask_layer, overlay_alpha, overlay, 1.0 - overlay_alpha, 0, overlay)

    banner_height = max(55, int(round(65 * scale_factor)))
    banner = np.zeros((banner_height, orig_w, 3), dtype=np.uint8)
    banner[:] = (25, 25, 25)

    header_text = f"AI Object Count: {total_idx} Objects  |  Model: FastSAM (Zero-Shot Instance Segmentation)"
    font_scale_hdr = max(0.55, 0.75 * scale_factor)
    cv2.putText(banner, header_text, (20, int(banner_height * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_hdr, (0, 255, 255), max(1, int(round(2 * scale_factor))), cv2.LINE_AA)

    final_annotated = np.vstack([banner, overlay])

    return {
        "count": total_idx,
        "items": items,
        "annotated_image": final_annotated,
        "overlay_image": overlay,
        "binary_mask": composite_mask,
        "original_image": img
    }


def evaluate_dataset_ai(dataset_dir, output_dir=None, conf=0.55):
    """Evaluates the AI model across an entire directory of images."""
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

    is_screws_benchmark = "screws" in dataset_dir.replace("\\", "/").lower() and "screwandbolt" not in dataset_dir.replace("\\", "/").lower()

    print(f"\n{'='*70}")
    print(f"AI EVALUATION: {os.path.basename(dataset_dir)} ({len(image_paths)} images)")
    print(f"Model: Pretrained FastSAM (SA-1B)  |  Confidence Threshold: {conf}")
    print(f"{'='*70}")
    if is_screws_benchmark:
        print(f"{'Filename':<30} | {'Detected Count':<16} | {'Ground Truth':<14} | {'Accuracy':<10}")
    else:
        print(f"{'Filename':<30} | {'Detected Count':<16} | {'Status':<10}")
    print(f"{'-'*70}")

    model, device = get_ai_model()
    results = []

    for img_p in image_paths:
        fname = os.path.basename(img_p)
        res = detect_items_ai(img_p, model=model, conf=conf)
        results.append({
            "filename": fname,
            "count": res["count"]
        })

        if is_screws_benchmark:
            gt = 43
            acc = max(0.0, 1.0 - abs(res["count"] - gt) / gt) * 100.0
            print(f"{fname:<30} | {res['count']:<16} | {gt:<14} | {acc:.1f}%")
        else:
            print(f"{fname:<30} | {res['count']:<16} | {'OK':<10}")

        if output_dir:
            out_img = os.path.join(output_dir, f"ai_detected_{os.path.splitext(fname)[0]}.jpg")
            out_mask = os.path.join(output_dir, f"ai_mask_{os.path.splitext(fname)[0]}.png")
            cv2.imwrite(out_img, res["annotated_image"])
            cv2.imwrite(out_mask, res["binary_mask"])

    print(f"{'-'*70}")
    counts = [r["count"] for r in results]
    if is_screws_benchmark:
        mean_acc = np.mean([max(0.0, 1.0 - abs(c - 43) / 43) * 100.0 for c in counts])
        print(f"Summary: Images={len(results)} | Mean Count={np.mean(counts):.2f} / 43 | Mean Accuracy={mean_acc:.2f}% (Target: > 95%)")
    else:
        print(f"Summary: Images={len(results)} | Mean Detected Count={np.mean(counts):.1f}")
    print(f"{'='*70}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="AI Object Counting and Mask Overlay using FastSAM")
    parser.add_argument("--input", type=str, default="data/Screws", help="Path to image file or directory")
    parser.add_argument("--output_dir", type=str, default="Athira_PT/AI/output", help="Directory to save output visualizations")
    parser.add_argument("--conf", type=float, default=0.55, help="Confidence threshold")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        evaluate_dataset_ai(args.input, output_dir=args.output_dir, conf=args.conf)
    elif os.path.isfile(args.input):
        res = detect_items_ai(args.input, conf=args.conf)
        print(f"\nProcessed {args.input}: Detected Objects={res['count']}")
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
