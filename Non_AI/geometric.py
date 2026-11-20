from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def estimate_background(gray: np.ndarray) -> np.ndarray:
    """Estimate smooth background illumination to handle vignetting and uneven lighting."""
    h, w = gray.shape
    scale = 0.25 if max(h, w) > 1000 else 0.5
    small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    ksize = max(21, int(min(small_h, small_w) * 0.08) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg_small = cv2.GaussianBlur(bg_small, (21, 21), 0)
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)


def compute_foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Build a clean binary mask of dark objects on a light background."""
    bg = estimate_background(gray)
    # Objects are darker than the background
    diff = cv2.subtract(bg, gray)
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)

    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Floor threshold prevents amplifying background noise on clean/empty regions
    thresh_val = max(float(otsu_thresh), 18.0)
    _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def split_merged_contour(
    contour: np.ndarray,
    expected_area: float,
    img_shape: tuple[int, int],
) -> list[np.ndarray]:
    """Split an oversized contour (touching objects) using distance-transform peaks."""
    x, y, bw, bh = cv2.boundingRect(contour)
    pad = 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img_shape[1], x + bw + pad), min(img_shape[0], y + bh + pad)

    roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = contour - np.array([[[x0, y0]]], dtype=contour.dtype)
    cv2.drawContours(roi_mask, [shifted], -1, 255, thickness=cv2.FILLED)

    dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)
    max_dist = float(dist_smooth.max())
    if max_dist <= 0:
        return [contour]

    min_sep = max(5, int(0.45 * np.sqrt(expected_area)) | 1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_sep, min_sep))
    local_max = (dist_smooth == cv2.dilate(dist_smooth, dilate_kernel)) & (
        dist_smooth >= 0.45 * max_dist
    )

    num_peaks, peak_labels, _, _ = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))
    if num_peaks <= 2:  # background + 1 peak -> cannot split further
        return [contour]

    # Assign each foreground pixel in ROI to its nearest peak via watershed on -dist
    markers = np.zeros(roi_mask.shape, dtype=np.int32)
    markers[roi_mask == 0] = 1
    markers[peak_labels > 0] = peak_labels[peak_labels > 0] + 1

    dist_norm = cv2.normalize(dist_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv_dist_bgr = cv2.cvtColor(255 - dist_norm, cv2.COLOR_GRAY2BGR)
    cv2.watershed(inv_dist_bgr, markers)

    sub_contours: list[np.ndarray] = []
    min_sub_area = 0.25 * expected_area
    for label in range(2, num_peaks + 1):
        sub_mask = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for sub_cnt in cnts:
            if cv2.contourArea(sub_cnt) >= min_sub_area:
                sub_contours.append(sub_cnt + np.array([[[x0, y0]]], dtype=sub_cnt.dtype))

    return sub_contours if sub_contours else [contour]


def detect_geometric_features(mask: np.ndarray) -> tuple[list[dict], str]:
    """
    Detect objects using geometric features:
    - Circular inner-hole detection for ring-like parts (e.g. hex nuts)
    - Contour geometry (area, circularity, oriented bounding box) with peak-based
      splitting for elongated/solid parts (e.g. screws)
    """
    h, w = mask.shape
    min_abs_area = max(20.0, (h * w) * 0.000015)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return [], "none"

    hierarchy = hierarchy[0]
    outer_indices = [i for i, h_info in enumerate(hierarchy) if h_info[3] == -1]
    hole_indices = [i for i, h_info in enumerate(hierarchy) if h_info[3] != -1]

    # Evaluate circular inner holes (strong geometric signature of nuts/washers)
    valid_holes: list[tuple[int, np.ndarray, float]] = []
    for idx in hole_indices:
        cnt = contours[idx]
        area = cv2.contourArea(cnt)
        if area < min_abs_area * 0.5:
            continue
        if circularity(cnt) >= 0.55:
            valid_holes.append((idx, cnt, area))

    # Filter holes by consistency around median hole area
    if len(valid_holes) >= 5:
        hole_areas = np.array([a for _, _, a in valid_holes])
        med_hole = float(np.median(hole_areas))
        consistent_holes = [
            (idx, cnt, area)
            for idx, cnt, area in valid_holes
            if 0.25 * med_hole <= area <= 3.0 * med_hole
        ]
    else:
        consistent_holes = []

    valid_outer = [
        (i, contours[i], cv2.contourArea(contours[i]))
        for i in outer_indices
        if cv2.contourArea(contours[i]) >= min_abs_area
    ]

    # Mode 1: Ring / hex-nut scene (dominated by consistent circular inner holes)
    if len(consistent_holes) >= 5 and len(consistent_holes) >= 0.5 * max(1, len(valid_outer)):
        detections: list[dict] = []
        for _, hole_cnt, _ in consistent_holes:
            (cx, cy), radius = cv2.minEnclosingCircle(hole_cnt)
            outer_radius = radius * 2.15
            detections.append(
                {
                    "kind": "circle",
                    "center": (int(round(cx)), int(round(cy))),
                    "inner_radius": max(2, int(round(radius))),
                    "outer_radius": max(4, int(round(outer_radius))),
                    "contour": hole_cnt,
                }
            )
        return detections, "ring_holes"

    # Mode 2: Solid / elongated parts (screws, bolts, etc.)
    if not valid_outer:
        return [], "contours"

    areas = np.array([a for _, _, a in valid_outer])
    median_area = float(np.median(areas))
    min_obj_area = max(min_abs_area, 0.25 * median_area)

    detections = []
    for _, cnt, area in valid_outer:
        if area < min_obj_area:
            continue

        parts = (
            split_merged_contour(cnt, median_area, (h, w))
            if area > 1.7 * median_area
            else [cnt]
        )

        for part in parts:
            part_area = cv2.contourArea(part)
            if part_area < min_obj_area:
                continue

            moments = cv2.moments(part)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                x, y, bw, bh = cv2.boundingRect(part)
                cx, cy = x + bw // 2, y + bh // 2

            circ = circularity(part)
            rect = cv2.minAreaRect(part)
            box = np.int32(cv2.boxPoints(rect))
            (_, _), radius = cv2.minEnclosingCircle(part)

            detections.append(
                {
                    "kind": "circle" if circ >= 0.78 else "obb",
                    "center": (cx, cy),
                    "outer_radius": max(4, int(round(radius))),
                    "box": box,
                    "contour": part,
                }
            )

    return detections, "contours"


def draw_overlay(img: np.ndarray, detections: list[dict], mode: str) -> np.ndarray:
    """Render translucent colored geometric overlays, labels, and total count banner."""
    h, w = img.shape[:2]
    scale = max(h, w) / 1500.0
    line_thickness = max(1, int(round(2 * scale)))
    font_scale = max(0.45, 0.55 * scale)
    font_thickness = max(1, int(round(1.6 * scale)))

    color_mask = np.zeros_like(img)
    rng = np.random.default_rng(seed=42)

    colors: list[tuple[int, int, int]] = []
    for det in detections:
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        colors.append(color)
        if mode == "ring_holes":
            cv2.circle(color_mask, det["center"], det["outer_radius"], color, thickness=cv2.FILLED)
            cv2.circle(color_mask, det["center"], det["inner_radius"], (0, 0, 0), thickness=cv2.FILLED)
        else:
            cv2.drawContours(color_mask, [det["contour"]], -1, color, thickness=cv2.FILLED)

    # Blend colored regions with the original image (60% original, 40% tint)
    overlay = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)

    # Draw crisp geometric boundaries and centroid indices
    for idx, (det, color) in enumerate(zip(detections, colors), start=1):
        cx, cy = det["center"]
        if det["kind"] == "circle":
            cv2.circle(overlay, (cx, cy), det["outer_radius"], (0, 0, 255), line_thickness, cv2.LINE_AA)
            if "inner_radius" in det:
                cv2.circle(overlay, (cx, cy), det["inner_radius"], color, line_thickness, cv2.LINE_AA)
        else:
            cv2.drawContours(overlay, [det["contour"]], -1, (0, 0, 255), line_thickness, cv2.LINE_AA)
            cv2.drawContours(overlay, [det["box"]], 0, color, max(1, line_thickness - 1), cv2.LINE_AA)

        label = f"#{idx}"
        label_scale = font_scale * (0.7 if len(detections) > 100 else 1.0)
        cv2.putText(
            overlay,
            label,
            (cx - int(10 * scale), cy + int(5 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    # Draw total detection count badge in the top-left corner
    banner_text = f"Count: {len(detections)}"
    banner_scale = max(0.8, 1.1 * scale)
    banner_thickness = max(2, int(round(2.5 * scale)))
    (tw, th), baseline = cv2.getTextSize(
        banner_text, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thickness
    )
    pad = int(14 * scale)
    cv2.rectangle(
        overlay,
        (pad, pad),
        (pad * 3 + tw, pad * 3 + th + baseline),
        (0, 0, 0),
        thickness=cv2.FILLED,
    )
    cv2.putText(
        overlay,
        banner_text,
        (pad * 2, pad * 2 + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        banner_scale,
        (255, 255, 255),
        banner_thickness,
        cv2.LINE_AA,
    )

    return overlay


def geometric_overlay(image_path: str | Path, output_path: str | Path = "geometric_result.png"):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = compute_foreground_mask(gray)
    detections, mode = detect_geometric_features(mask)
    overlay = draw_overlay(img, detections, mode)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    print(f"Detected {len(detections)} objects ({mode}). Saved overlay to {output_path}")


def process_images(images_dir: str | Path, output_dir: str | Path):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:
        output_path = output_dir / f"{image_path.stem}.png"
        geometric_overlay(image_path, output_path)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    process_images(script_dir.parent / "images", script_dir / "geometric")
