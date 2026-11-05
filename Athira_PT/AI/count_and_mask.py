"""
AI Item Counting and Mask Overlay Pipeline (Dual-Class: Screws & Bolts)
Author: Athira PT
Task: Count screws and bolts in images and overlay instance masks using Deep Learning / Foundation Models.
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


def classify_mask(mask_binary, scale_factor=1.0):
    """
    Classify instance mask as 'Bolt' or 'Screw' based on physical dimensions:
    - Bolts have thick shanks (thickness >= 75px at native res) or large area (>= 8000px).
    - Screws are thinner and more elongated (thickness < 75px, area < 8000px).
    """
    area = np.sum(mask_binary > 0)
    cnts, _ = cv2.findContours(mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "Screw", 0.0, float(area)
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    thickness = min(rect[1])

    if thickness >= 75.0 or area >= 8000.0:
        return "Bolt", float(thickness), float(area)
    else:
        return "Screw", float(thickness), float(area)


def detect_items_ai(
    image_input,
    model=None,
    model_type="fastsam",
    conf=0.55,
    iou=0.85,
    min_mask_area=80,
    overlay_alpha=0.45,
    imgsz=1024
):
    """
    Detects, segments, classifies (Screw vs Bolt), and counts items using AI.

    Returns:
        dict containing:
            - 'count': Total detected count
            - 'bolt_count': Number of bolts
            - 'screw_count': Number of screws
            - 'items': List of item metadata (id, category, bbox, centroid, area)
            - 'annotated_image': Visual overlay with labels and summary banner
            - 'overlay_image': Overlay without banner
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

    if model is None:
        model, device, model_type = get_ai_model(model_type)
    else:
        device = 0 if torch.cuda.is_available() else "cpu"

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

    composite_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    mask_layer = np.zeros_like(img)
    overlay = img.copy()

    items = []
    bolt_idx = 0
    screw_idx = 0
    total_idx = 0

    COLOR_SCREW = (0, 255, 0)     # Bright Green
    COLOR_BOLT  = (0, 165, 255)   # Vibrant Orange / Gold

    scale_factor = max(orig_h, orig_w) / 1080.0

    if res.masks is not None and len(res.masks) > 0:
        masks_tensor = res.masks.data
        for idx in range(len(masks_tensor)):
            m_np = masks_tensor[idx].cpu().numpy().astype(np.uint8)
            
            if m_np.shape[:2] != (orig_h, orig_w):
                m_np = cv2.resize(m_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            area = np.sum(m_np > 0)
            if area < min_mask_area:
                continue

            category, thickness, _ = classify_mask(m_np, scale_factor=scale_factor)

            total_idx += 1
            if category == "Bolt":
                bolt_idx += 1
                color = COLOR_BOLT
                label = f"Bolt #{bolt_idx}"
            else:
                screw_idx += 1
                color = COLOR_SCREW
                label = f"Screw #{screw_idx}"

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

            font_scale = max(0.35, 0.5 * scale_factor)
            thickness_txt = max(1, int(round(1.5 * scale_factor)))
            cv2.putText(overlay, label, (x, max(18, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness_txt, cv2.LINE_AA)

            items.append({
                "id": total_idx,
                "category": category,
                "bbox": [int(x), int(y), int(w), int(h)],
                "centroid": (int(cx), int(cy)),
                "area": int(area)
            })

    cv2.addWeighted(mask_layer, overlay_alpha, overlay, 1.0 - overlay_alpha, 0, overlay)

    banner_height = max(55, int(round(65 * scale_factor)))
    banner = np.zeros((banner_height, orig_w, 3), dtype=np.uint8)
    banner[:] = (25, 25, 25)
    
    header_text = f"AI Count: {total_idx} Total  |  Bolts: {bolt_idx} (Orange)  |  Screws: {screw_idx} (Green)  |  Model: {model_type.upper()}"
    font_scale_hdr = max(0.55, 0.75 * scale_factor)
    cv2.putText(banner, header_text, (20, int(banner_height * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_hdr, (0, 255, 255), max(1, int(round(2 * scale_factor))), cv2.LINE_AA)

    final_annotated = np.vstack([banner, overlay])

    return {
        "count": total_idx,
        "bolt_count": bolt_idx,
        "screw_count": screw_idx,
        "items": items,
        "annotated_image": final_annotated,
        "overlay_image": overlay,
        "binary_mask": composite_mask,
        "original_image": img
    }


def evaluate_dataset_ai(dataset_dir, output_dir=None, model_type="fastsam", conf=0.55):
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

    print(f"\n{'='*75}")
    print(f"AI EVALUATION ON DATASET: {os.path.basename(dataset_dir)} ({len(image_paths)} images)")
    print(f"Model: {model_type.upper()}  |  Confidence: {conf}")
    print(f"{'='*75}")
    print(f"{'Filename':<30} | {'Total Count':<12} | {'Bolts':<8} | {'Screws':<8} | {'Status':<8}")
    print(f"{'-'*75}")

    model, device, m_type = get_ai_model(model_type)

    results = []
    for img_p in image_paths:
        fname = os.path.basename(img_p)
        res = detect_items_ai(img_p, model=model, model_type=m_type, conf=conf)
        results.append({
            "filename": fname,
            "count": res["count"],
            "bolt_count": res["bolt_count"],
            "screw_count": res["screw_count"]
        })
        print(f"{fname:<30} | {res['count']:<12} | {res['bolt_count']:<8} | {res['screw_count']:<8} | {'OK':<8}")

        if output_dir:
            out_img = os.path.join(output_dir, f"ai_detected_{os.path.splitext(fname)[0]}.jpg")
            out_mask = os.path.join(output_dir, f"ai_mask_{os.path.splitext(fname)[0]}.png")
            cv2.imwrite(out_img, res["annotated_image"])
            cv2.imwrite(out_mask, res["binary_mask"])

    print(f"{'-'*75}")
    counts = [r["count"] for r in results]
    bolts = [r["bolt_count"] for r in results]
    screws = [r["screw_count"] for r in results]
    print(f"Summary: Images={len(results)} | Mean Total={np.mean(counts):.1f} | Mean Bolts={np.mean(bolts):.1f} | Mean Screws={np.mean(screws):.1f}")
    print(f"{'='*75}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="AI Screws & Bolts Counting and Mask Overlay using Deep Learning")
    parser.add_argument("--input", type=str, default="data/Screws", help="Path to image file or directory")
    parser.add_argument("--output_dir", type=str, default="Athira_PT/AI/output", help="Directory to save output visualizations")
    parser.add_argument("--model", type=str, default="fastsam", choices=["fastsam", "custom"], help="AI model to use")
    parser.add_argument("--conf", type=float, default=0.55, help="Confidence threshold")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        evaluate_dataset_ai(args.input, output_dir=args.output_dir, model_type=args.model, conf=args.conf)
    elif os.path.isfile(args.input):
        res = detect_items_ai(args.input, model_type=args.model, conf=args.conf)
        print(f"\nProcessed {args.input}: Total={res['count']}, Bolts={res['bolt_count']}, Screws={res['screw_count']}")
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
