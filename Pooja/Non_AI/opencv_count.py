#!/usr/bin/env python3
"""
opencv_count.py
Non-AI counting + mask overlay for screws/bolts.

Usage:
    python opencv_count.py --input /path/to/images --output /path/to/results
"""

import cv2
import numpy as np
import os
import argparse
import json
from tqdm import tqdm


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def preprocess(img):
    """Grayscale -> CLAHE -> Top-hat contrast to emphasize objects."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # large kernel top-hat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return tophat


def segment(img):
    """Segment likely screw regions. Return binary mask (uint8 0/255)."""
    pre = preprocess(img)
    blur = cv2.GaussianBlur(pre, (5, 5), 0)

    # Adaptive threshold - tuned for these images
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, -8
    )

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # remove small specks
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(clean)
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 20:
            cv2.drawContours(mask, [c], -1, 255, -1)

    return mask


def separate_objects_watershed(img, mask):
    """
    Use distance transform + watershed to split touching objects.
    Returns labeled markers (int32) where each object has distinct label (>1).
    """
    # sure background
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=2)

    # distance transform for foreground
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # threshold to get sure foreground
    ret, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # markers
    ret, markers = cv2.connectedComponents(sure_fg)
    # add one to all labels so that background is 1
    markers = markers + 1
    # mark unknown as 0
    markers[unknown == 255] = 0

    # perform watershed on color image
    img_color = img.copy()
    cv2.watershed(img_color, markers)
    # after watershed, boundary pixels are set to -1
    # convert markers such that each region id >=2 is an object
    return markers


def postprocess_and_save(img, mask, out_overlay_path, out_masks_dir, basename):
    """
    Find contours from mask or markers and save per-object masks and overlay.
    Returns count and objects list.
    """
    # Use watershed if objects overlap significantly
    # If average object area is large compared to image, skip watershed to save time
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours seem merged (a few contours but expected many), try watershed
    use_watershed = False
    if len(cnts) < 8:  # heuristic for these datasets; adjust if needed
        # check if average contour area is very large
        areas = [cv2.contourArea(c) for c in cnts] if cnts else [0]
        if len(areas) and np.mean(areas) > 2000:
            use_watershed = True

    objects = []
    mask_all = np.zeros(img.shape[:2], dtype=np.uint8)

    if use_watershed:
        markers = separate_objects_watershed(img, mask)
        # markers >=2 are objects; loop through them
        unique_labels = np.unique(markers)
        idx = 0
        for lab in unique_labels:
            if lab <= 1:
                continue
            obj_mask = np.zeros_like(mask)
            obj_mask[markers == lab] = 255
            # filter small
            cnts_obj, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts_obj:
                continue
            c = max(cnts_obj, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area < 30:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / (h + 1e-6)
            if aspect > 8 or aspect < 0.12:
                continue
            idx += 1
            objects.append({'id': idx, 'area': int(area), 'bbox': [int(x), int(y), int(w), int(h)]})
            cv2.imwrite(os.path.join(out_masks_dir, f"{basename}_mask_{idx:03d}.png"), obj_mask)
            mask_all = cv2.bitwise_or(mask_all, obj_mask)
    else:
        idx = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 30:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / (h + 1e-6)
            if aspect > 8 or aspect < 0.12:
                continue
            obj_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(obj_mask, [c], -1, 255, -1)
            idx += 1
            objects.append({'id': idx, 'area': int(area), 'bbox': [int(x), int(y), int(w), int(h)]})
            cv2.imwrite(os.path.join(out_masks_dir, f"{basename}_mask_{idx:03d}.png"), obj_mask)
            mask_all = cv2.bitwise_or(mask_all, obj_mask)

    # create overlay (green highlight)
    overlay = img.copy()
    overlay[mask_all == 255] = (0, 255, 0)
    alpha = 0.5
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imwrite(out_overlay_path, out)

    return len(objects), objects


def process_image(path, out_root):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Unable to read image: {path}")
    basename = os.path.splitext(os.path.basename(path))[0]
    masks_dir = os.path.join(out_root, "masks")
    overlays_dir = os.path.join(out_root, "overlays")
    ensure_dir(masks_dir)
    ensure_dir(overlays_dir)

    mask = segment(img)
    count, objects = postprocess_and_save(img, mask, os.path.join(overlays_dir, f"{basename}.png"), masks_dir, basename)

    return basename, count, objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder with images")
    parser.add_argument("--output", required=True, help="Output folder to save overlays, masks, counts")
    args = parser.parse_args()

    ensure_dir(args.output)
    ensure_dir(os.path.join(args.output, "masks"))
    ensure_dir(os.path.join(args.output, "overlays"))

    results = {}
    files = sorted([f for f in os.listdir(args.input) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    for f in tqdm(files, desc="Processing images"):
        path = os.path.join(args.input, f)
        try:
            basename, count, objects = process_image(path, args.output)
            results[f] = {"count": count, "objects": objects}
        except Exception as e:
            print(f"Error processing {f}: {e}")
            results[f] = {"count": 0, "error": str(e)}

    out_json = os.path.join(args.output, "counts.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
