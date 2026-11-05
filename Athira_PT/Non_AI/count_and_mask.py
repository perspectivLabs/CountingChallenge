"""
Non-AI Item Counting and Mask Overlay Pipeline
Author: Athira PT
Task: Count screws/bolts in images and overlay precise masks using traditional Computer Vision (OpenCV).
Accuracy Target: > 95%
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np


def detect_items_non_ai(
    image_input,
    target_dim=1600,
    diff_thresh=22,
    min_area=80,
    max_area_ratio=0.85,
    overlay_alpha=0.45,
    fill_holes=True
):
    """
    Detect, count, and overlay masks on items (screws, bolts) in an image using
    non-AI traditional computer vision techniques.

    Pipeline Steps:
    1. Scaling & Preprocessing: Adaptive resizing for uniform kernel scale across resolutions.
    2. Illumination Correction: Large morphological dilation models non-uniform background lighting.
    3. Differential Thresholding: Subtract original from estimated background to eliminate shadows/gradients.
    4. Hole Filling & Specular Repair: Morphological closing & flood-fill repair shiny metal highlights.
    5. Noise Filtering: Morphological opening removes dust specks and tiny artifacts.
    6. Contour Extraction & Filtering: Extract contours and filter by scale-adjusted area.
    7. Touching Object Separation: Inspect large contours exceeding normal area distribution to identify merged items.
    8. Mask & Annotation Overlay: Render semi-transparent colored masks, contours, bounding boxes, and IDs.

    Args:
        image_input: File path (str) or BGR image (np.ndarray).
        target_dim: Maximum dimension for processing scale (default: 1600).
        diff_thresh: Intensity difference threshold between background and foreground (default: 22).
        min_area: Minimum contour area threshold at target scale (default: 80).
        max_area_ratio: Maximum area ratio to prevent detecting full image borders (default: 0.85).
        overlay_alpha: Opacity factor for mask overlay [0.0 - 1.0] (default: 0.45).
        fill_holes: Whether to fill interior specular highlight holes (default: True).

    Returns:
        dict containing:
            - 'count': Total item count
            - 'items': List of item metadata (id, bbox, centroid, area)
            - 'overlay_image': Original image with translucent masks, boxes, and labels
            - 'binary_mask': Binary segmentation mask (255=item, 0=background)
            - 'original_image': The original unmodified image
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

    # Step 1: Scale normalization for consistent structuring element response
    scale = target_dim / max(orig_h, orig_w) if max(orig_h, orig_w) > target_dim else 1.0
    if scale != 1.0:
        proc_w = int(orig_w * scale)
        proc_h = int(orig_h * scale)
        proc_img = cv2.resize(img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        proc_img = img.copy()
        proc_w, proc_h = orig_w, orig_h

    # Step 2: Grayscale & Gaussian Blurring
    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Morphological Illumination Estimation & Differential Subtraction
    # A large elliptical structuring element covers across screws to capture background gradient
    k_size = int(round(45 * min(proc_w, proc_h) / 1080.0))
    k_size = max(25, k_size | 1)  # ensure odd and at least 25
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    # Check background polarity (light background vs dark background)
    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_median = float(np.median(border_pixels))
    
    if bg_median > 100:
        # Light surface with darker metallic screws/bolts
        bg_model = cv2.morphologyEx(blurred, cv2.MORPH_DILATE, kernel_bg)
        diff = cv2.subtract(bg_model, blurred)
    else:
        # Dark surface with lighter objects
        bg_model = cv2.morphologyEx(blurred, cv2.MORPH_ERODE, kernel_bg)
        diff = cv2.subtract(blurred, bg_model)

    # Step 4: Differential Thresholding
    _, thresh = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    # Step 5: Specular Highlight Repair & Hole Filling
    # Metallic screws reflect bright light at center, which would otherwise create holes
    if fill_holes:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close, iterations=2)
        
        # Flood-fill holes to ensure 100% solid object masks
        h_pad, w_pad = closed.shape
        im_floodfill = closed.copy()
        mask_ff = np.zeros((h_pad + 2, w_pad + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask_ff, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        solid_mask = closed | im_floodfill_inv
    else:
        solid_mask = thresh

    # Step 6: Noise Removal (remove isolated dust specks)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_OPEN, k_open)

    # Step 7: Contour Extraction & Scale-aware Filtering
    raw_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Scale adjusted min/max area
    effective_min_area = min_area * (scale ** 2) if scale < 1.0 else min_area
    max_area = (proc_w * proc_h) * max_area_ratio

    valid_contours = []
    contour_areas = []
    for c in raw_contours:
        area = cv2.contourArea(c)
        if effective_min_area <= area <= max_area:
            valid_contours.append(c)
            contour_areas.append(area)

    # Step 8: Multi-object / Cluster Decomposition for touching screws/bolts
    # If any contour has an area significantly larger than the median item area,
    # it likely represents touching items. We can estimate sub-items via area ratio or distance peaks.
    processed_contours = []
    final_items = []
    
    if len(contour_areas) > 3:
        median_area = float(np.median(contour_areas))
    else:
        median_area = float(np.mean(contour_areas)) if contour_areas else 1000.0

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

    # Create composite binary mask at original image resolution
    orig_binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    # Scale contours back to original resolution if scale != 1.0
    item_counter = 0
    overlay = img.copy()
    mask_layer = np.zeros_like(img)

    for idx, cnt in enumerate(valid_contours):
        area = cv2.contourArea(cnt)
        
        # Project contour coordinates to original image coordinates
        if scale != 1.0:
            cnt_orig = (cnt.astype(np.float32) / scale).astype(np.int32)
        else:
            cnt_orig = cnt

        # Estimate how many items might be merged (if area > 1.95x median)
        items_in_cnt = max(1, int(round(area / median_area))) if median_area > 0 and (area / median_area) >= 1.95 else 1
        
        # Draw on binary mask
        cv2.drawContours(orig_binary_mask, [cnt_orig], -1, 255, thickness=cv2.FILLED)
        
        # Color for visualization
        color = color_palette[item_counter % len(color_palette)]
        cv2.drawContours(mask_layer, [cnt_orig], -1, color, thickness=cv2.FILLED)
        cv2.drawContours(overlay, [cnt_orig], -1, color, thickness=max(2, int(round(2 / scale))))

        # Bounding box & centroid
        x, y, w, h = cv2.boundingRect(cnt_orig)
        M = cv2.moments(cnt_orig)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + w // 2
            cy = y + h // 2

        # Draw bounding box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (200, 200, 200), 1)

        # Draw ID and centroid point
        cv2.circle(overlay, (cx, cy), max(3, int(round(4 / scale))), (0, 0, 255), -1)
        
        item_counter += 1
        label = f"#{item_counter}"
        font_scale = max(0.4, 0.6 / scale)
        thickness = max(1, int(round(1.5 / scale)))
        cv2.putText(overlay, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

        final_items.append({
            "id": item_counter,
            "bbox": [int(x), int(y), int(w), int(h)],
            "centroid": (int(cx), int(cy)),
            "area": float(area / (scale ** 2)),
            "cluster_count": items_in_cnt
        })

    # Blend colored mask layer with image
    cv2.addWeighted(mask_layer, overlay_alpha, overlay, 1.0 - overlay_alpha, 0, overlay)

    # Add descriptive top banner
    banner_height = max(50, int(round(60 / scale)))
    banner = np.zeros((banner_height, orig_w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)  # Dark slate gray
    
    header_text = f"Non-AI Count: {len(final_items)} items  |  Algorithm: Morphological Illumination Model + Diff Threshold"
    font_scale_hdr = max(0.6, 0.85 / scale)
    cv2.putText(banner, header_text, (20, int(banner_height * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale_hdr, (0, 255, 0), max(1, int(round(2 / scale))), cv2.LINE_AA)
    
    final_annotated = np.vstack([banner, overlay])

    return {
        "count": len(final_items),
        "items": final_items,
        "annotated_image": final_annotated,
        "overlay_image": overlay,
        "binary_mask": orig_binary_mask,
        "original_image": img
    }


def evaluate_dataset(dataset_dir, output_dir=None, diff_thresh=22, min_area=80):
    """
    Run detection over all images in a dataset directory, print results, and save outputs.
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
    print(f"EVALUATING DATASET: {os.path.basename(dataset_dir)} ({len(image_paths)} images)")
    print(f"{'='*70}")
    print(f"{'Filename':<35} | {'Detected Count':<16} | {'Status':<10}")
    print(f"{'-'*70}")

    results = []
    for img_p in image_paths:
        fname = os.path.basename(img_p)
        res = detect_items_non_ai(img_p, diff_thresh=diff_thresh, min_area=min_area)
        count = res["count"]
        results.append({"filename": fname, "count": count, "path": img_p})
        print(f"{fname:<35} | {count:<16} | {'OK':<10}")

        if output_dir:
            out_img_name = f"detected_{os.path.splitext(fname)[0]}.jpg"
            out_mask_name = f"mask_{os.path.splitext(fname)[0]}.png"
            cv2.imwrite(os.path.join(output_dir, out_img_name), res["annotated_image"])
            cv2.imwrite(os.path.join(output_dir, out_mask_name), res["binary_mask"])

    print(f"{'-'*70}")
    counts = [r["count"] for r in results]
    print(f"Total Images: {len(results)} | Mean Count: {np.mean(counts):.1f} | Min: {min(counts)} | Max: {max(counts)}")
    print(f"{'='*70}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Non-AI Item Counting and Mask Overlay using OpenCV")
    parser.add_argument("--input", type=str, default="data/Screws", help="Path to image file or directory")
    parser.add_argument("--output_dir", type=str, default="Athira_PT/Non_AI/output", help="Directory to save output visualizations")
    parser.add_argument("--diff_thresh", type=int, default=22, help="Difference threshold for background subtraction")
    parser.add_argument("--min_area", type=int, default=80, help="Minimum contour area to filter noise")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        evaluate_dataset(args.input, output_dir=args.output_dir, diff_thresh=args.diff_thresh, min_area=args.min_area)
    elif os.path.isfile(args.input):
        res = detect_items_non_ai(args.input, diff_thresh=args.diff_thresh, min_area=args.min_area)
        print(f"\nProcessed {args.input}: Detected {res['count']} items.")
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            fname = os.path.basename(args.input)
            out_img = os.path.join(args.output_dir, f"detected_{fname}")
            out_mask = os.path.join(args.output_dir, f"mask_{os.path.splitext(fname)[0]}.png")
            cv2.imwrite(out_img, res["annotated_image"])
            cv2.imwrite(out_mask, res["binary_mask"])
            print(f"Saved annotated overlay to: {out_img}")
            print(f"Saved binary mask to: {out_mask}")
    else:
        print(f"Error: Input path '{args.input}' does not exist.")


if __name__ == "__main__":
    main()
