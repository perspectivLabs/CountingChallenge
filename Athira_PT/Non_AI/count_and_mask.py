"""
Non-AI Item Counting and Mask Overlay Pipeline (Dual-Class: Screws & Bolts)
Author: Athira PT
Task: Count screws and bolts in images and overlay precise masks using traditional Computer Vision (OpenCV).
Accuracy Target: > 95%
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np


def classify_fastener(area_norm, thickness_norm):
    """
    Classify a fastener as 'Bolt' or 'Screw' based on physical geometry:
    - Bolts have thick shanks (thickness >= 65px at scale 1600) or large hex heads/area (>= 3500px).
    - Screws have thinner shanks and smaller area (thickness < 65px, area < 3500px).
    """
    if thickness_norm >= 65.0 or area_norm >= 3500.0:
        return "Bolt"
    else:
        return "Screw"


def segment_dense_cluster(roi_mask, offset_xy, scale=1.0):
    """
    Decomposes a dense cluster of touching fasteners using Distance Transform
    local peak detection and watershed segmentation.
    """
    x_off, y_off = offset_xy
    dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    
    # Dynamic peak threshold based on maximum distance inside the cluster
    k_size = 9
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(dist, kernel)
    
    # Local maxima at least 6px from boundaries and 25% of peak
    peaks = (dist == dilated) & (dist > max(6.0, 0.22 * dist.max()))
    
    _, markers = cv2.connectedComponents(peaks.astype(np.uint8))
    num_peaks = np.max(markers)
    
    if num_peaks <= 1:
        cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c + [x_off, y_off] for c in cnts if cv2.contourArea(c) > 30]

    # Watershed on inverted distance map
    roi_bgr = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    sure_bg = cv2.dilate(roi_mask, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, (peaks * 255).astype(np.uint8))
    markers = markers + 1
    markers[unknown == 255] = 0
    
    cv2.watershed(roi_bgr, markers)
    
    sub_contours = []
    for mid in range(2, num_peaks + 2):
        sub_mask = (markers == mid).astype(np.uint8)
        c_sub, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cs in c_sub:
            if cv2.contourArea(cs) > 35:
                sub_contours.append(cs + [x_off, y_off])
                
    return sub_contours if len(sub_contours) > 1 else [cnts[0] + [x_off, y_off]]


def detect_items_non_ai(
    image_input,
    target_dim=1600,
    overlay_alpha=0.45,
    min_area=45
):
    """
    Detect, count, classify (Screw vs Bolt), and overlay masks using OpenCV.

    Pipeline Steps:
    1. Scale Normalization: Consistent resolution (target_dim 1600 px).
    2. Illumination Modeling: Morphological background subtraction removes lighting gradients.
    3. Differential Thresholding: Separates metallic objects from surface.
    4. Contour-Level Solidification: Fills specular highlight holes inside each object.
    5. Cluster Decomposition: Resolves dense clusters of touching bolts into individual items.
    6. Dual-Class Classification: Classifies each item as 'Bolt' (thick/heavy) vs 'Screw' (thin/elongated).
    7. Color-Coded Overlay: Vibrant Orange for Bolts, Bright Green for Screws.

    Returns:
        dict containing:
            - 'count': Total item count
            - 'bolt_count': Number of bolts detected
            - 'screw_count': Number of screws detected
            - 'items': List of item metadata (id, category, bbox, centroid, area)
            - 'annotated_image': Visual overlay with labels and summary banner
            - 'overlay_image': Overlay without banner
            - 'binary_mask': Composite binary mask
            - 'original_image': Input image
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

    # Step 1: Scale normalization
    scale = target_dim / max(orig_h, orig_w) if max(orig_h, orig_w) > target_dim else 1.0
    if scale != 1.0:
        proc_w = int(orig_w * scale)
        proc_h = int(orig_h * scale)
        proc_img = cv2.resize(img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        proc_img = img.copy()
        proc_w, proc_h = orig_w, orig_h

    # Step 2: Grayscale & Gaussian blur
    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Background illumination estimation
    k_size = int(round(51 * min(proc_w, proc_h) / 1080.0))
    k_size = max(31, k_size | 1)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    bg_model = cv2.morphologyEx(blurred, cv2.MORPH_DILATE, kernel_bg)
    diff = cv2.subtract(bg_model, blurred)

    # Step 4: Differential Thresholding
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Step 5: Contour-level hole filling (NO global floodfill to prevent cluster bridging)
    raw_cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(closed)
    for c in raw_cnts:
        if cv2.contourArea(c) > 30:
            cv2.drawContours(solid_mask, [c], -1, 255, thickness=cv2.FILLED)

    cleaned = cv2.morphologyEx(solid_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    pre_cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Estimate median single-item area
    valid_areas = [cv2.contourArea(c) for c in pre_cnts if 100 < cv2.contourArea(c) < 5000]
    median_a = float(np.median(valid_areas)) if valid_areas else 1200.0

    # Step 6: Cluster decomposition for touching bolt/screw clusters
    all_contours = []
    for c in pre_cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if area > median_a * 2.2:
            # Touching cluster: decompose
            x, y, w, h = cv2.boundingRect(c)
            roi = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(roi, [c - [x, y]], -1, 255, thickness=cv2.FILLED)
            subs = segment_dense_cluster(roi, (x, y), scale=scale)
            all_contours.extend(subs)
        else:
            all_contours.append(c)

    # Step 7: Classification & Visualization Overlay
    orig_binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    overlay = img.copy()
    mask_layer = np.zeros_like(img)

    COLOR_SCREW = (0, 255, 0)     # Bright Green
    COLOR_BOLT  = (0, 165, 255)   # Vibrant Orange / Gold

    items = []
    bolt_idx = 0
    screw_idx = 0
    total_idx = 0

    scale_factor = max(orig_h, orig_w) / 1080.0

    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(cnt)
        thickness = min(rect[1]) / scale
        norm_area = area / (scale ** 2)

        category = classify_fastener(norm_area, thickness)

        if scale != 1.0:
            cnt_orig = (cnt.astype(np.float32) / scale).astype(np.int32)
        else:
            cnt_orig = cnt

        total_idx += 1
        if category == "Bolt":
            bolt_idx += 1
            color = COLOR_BOLT
            label = f"Bolt #{bolt_idx}"
        else:
            screw_idx += 1
            color = COLOR_SCREW
            label = f"Screw #{screw_idx}"

        cv2.drawContours(orig_binary_mask, [cnt_orig], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask_layer, [cnt_orig], -1, color, thickness=cv2.FILLED)
        cv2.drawContours(overlay, [cnt_orig], -1, color, thickness=max(2, int(round(2 * scale_factor))))

        x, y, w, h = cv2.boundingRect(cnt_orig)
        M = cv2.moments(cnt_orig)
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
            "area": float(norm_area)
        })

    cv2.addWeighted(mask_layer, overlay_alpha, overlay, 1.0 - overlay_alpha, 0, overlay)

    banner_height = max(55, int(round(65 * scale_factor)))
    banner = np.zeros((banner_height, orig_w, 3), dtype=np.uint8)
    banner[:] = (25, 25, 25)

    header_text = f"Non-AI Count: {total_idx} Total  |  Bolts: {bolt_idx} (Orange)  |  Screws: {screw_idx} (Green)"
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
        "binary_mask": orig_binary_mask,
        "original_image": img
    }


def evaluate_dataset(dataset_dir, output_dir=None):
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
    print(f"NON-AI EVALUATION: {os.path.basename(dataset_dir)} ({len(image_paths)} images)")
    print(f"{'='*75}")
    print(f"{'Filename':<30} | {'Total Count':<12} | {'Bolts':<8} | {'Screws':<8} | {'Status':<8}")
    print(f"{'-'*75}")

    results = []
    for img_p in image_paths:
        fname = os.path.basename(img_p)
        res = detect_items_non_ai(img_p)
        results.append({
            "filename": fname,
            "count": res["count"],
            "bolt_count": res["bolt_count"],
            "screw_count": res["screw_count"]
        })
        print(f"{fname:<30} | {res['count']:<12} | {res['bolt_count']:<8} | {res['screw_count']:<8} | {'OK':<8}")

        if output_dir:
            out_img = os.path.join(output_dir, f"detected_{os.path.splitext(fname)[0]}.jpg")
            out_mask = os.path.join(output_dir, f"mask_{os.path.splitext(fname)[0]}.png")
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
    parser = argparse.ArgumentParser(description="Non-AI Screws & Bolts Counting and Mask Overlay using OpenCV")
    parser.add_argument("--input", type=str, default="data/Screws", help="Path to image file or directory")
    parser.add_argument("--output_dir", type=str, default="Athira_PT/Non_AI/output", help="Directory to save output visualizations")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        evaluate_dataset(args.input, output_dir=args.output_dir)
    elif os.path.isfile(args.input):
        res = detect_items_non_ai(args.input)
        print(f"\nProcessed {args.input}: Total={res['count']}, Bolts={res['bolt_count']}, Screws={res['screw_count']}")
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
