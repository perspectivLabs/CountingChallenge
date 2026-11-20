from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def watershed_overlay(image_path: str | Path, output_path: str | Path = "watershed_result.png"):
    # 1. Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Thresholding (Use THRESH_BINARY_INV if objects are darker than background;
    #    switch to THRESH_BINARY if objects are brighter than background)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Noise removal via morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. Sure background & sure foreground
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Adjust 0.5–0.7 depending on how heavily objects overlap:
    # Higher values separate more aggressive overlaps; lower values prevent splitting single objects
    _, sure_fg = cv2.threshold(dist_transform, 0.55 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 5. Unknown boundary region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Marker labelling
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Shift background from 0 to 1
    markers[unknown == 255] = 0  # Mark unknown border region as 0

    # 7. Run watershed
    markers = cv2.watershed(img, markers)

    # --- VISUAL OVERLAY CONSTRUCTION ---
    color_mask = np.zeros_like(img)
    rng = np.random.default_rng(seed=42)  # Fixed seed for consistent colors

    # Color each distinct segmented region (label > 1; label 1 is background, -1 is boundary)
    unique_labels = [lbl for lbl in np.unique(markers) if lbl > 1]
    for label in unique_labels:
        color = rng.integers(60, 255, size=3, dtype=np.uint8)
        color_mask[markers == label] = color

    # Blend colored regions with the original image (60% original, 40% tint)
    overlay = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)

    # Draw crisp red watershed boundary lines (excluding the 1px outer image border)
    inner_boundaries = np.zeros(markers.shape, dtype=bool)
    inner_boundaries[1:-1, 1:-1] = markers[1:-1, 1:-1] == -1
    overlay[inner_boundaries] = [0, 0, 255]  # Red in BGR

    # Optional: Label each segmented object at its centroid
    for idx, label in enumerate(unique_labels, start=1):
        y_coords, x_coords = np.where(markers == label)
        cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
        cv2.putText(
            overlay,
            f"#{idx}",
            (cx - 10, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Draw total detection count badge in the top-left corner
    h, w = overlay.shape[:2]
    scale = max(h, w) / 1500.0
    banner_text = f"Count: {len(unique_labels)}"
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    print(f"Detected {len(unique_labels)} objects. Saved overlay to {output_path}")


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
        watershed_overlay(image_path, output_path)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    process_images(script_dir.parent / "images", script_dir / "watershed")
