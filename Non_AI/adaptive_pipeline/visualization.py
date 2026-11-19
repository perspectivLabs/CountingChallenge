from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .features import ComponentFeatures
from .nuts import InstanceMask

PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, PNG_PARAMS)


def draw_corner_badge(
    img: np.ndarray,
    text: str,
    subtitle: str | None = None,
) -> np.ndarray:
    """Draw a clean resolution-scaled count badge in the top-left corner."""
    h, w = img.shape[:2]
    scale = max(h, w) / 1500.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    banner_scale = max(0.75, 1.05 * scale)
    banner_thick = max(2, int(round(2.4 * scale)))
    pad = max(10, int(round(14 * scale)))

    (tw, th), baseline = cv2.getTextSize(text, font, banner_scale, banner_thick)
    box_w = tw + pad * 2
    box_h = th + baseline + pad * 2

    sub_scale = max(0.50, 0.65 * scale)
    sub_thick = max(1, int(round(1.6 * scale)))
    sub_h = 0
    if subtitle:
        (stw, sth), sbase = cv2.getTextSize(subtitle, font, sub_scale, sub_thick)
        box_w = max(box_w, stw + pad * 2)
        sub_h = sth + sbase + int(pad * 0.6)
        box_h += sub_h

    cv2.rectangle(img, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(
        img,
        text,
        (pad * 2, pad * 2 + th),
        font,
        banner_scale,
        (255, 255, 255),
        banner_thick,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            img,
            subtitle,
            (pad * 2, pad * 2 + th + sub_h),
            font,
            sub_scale,
            (180, 230, 255),
            sub_thick,
            cv2.LINE_AA,
        )
    return img


def render_classification_overlay(
    img_bgr: np.ndarray,
    components: list[ComponentFeatures],
    scene_type: str,
) -> np.ndarray:
    """Render Stage 2 3-case geometric classification overlay with skeleton visualization."""
    h, w = img_bgr.shape[:2]
    scale = max(h, w) / 1500.0
    thick = max(1, int(round(2 * scale)))

    color_map = {
        "isolated_screw": (60, 210, 60),    # Green
        "isolated_nut": (60, 210, 60),      # Green
        "touching_nuts": (240, 140, 40),    # Blue-cyan
        "touching_screws": (50, 60, 240),   # Red
    }

    color_mask = np.zeros_like(img_bgr)
    counts = {"isolated": 0, "touching_nuts": 0, "touching_screws": 0}

    for comp in components:
        if comp.case_label == "noise":
            continue
        x0, y0, bw, bh = comp.bbox
        col = color_map.get(comp.case_label, (200, 200, 200))
        roi_color = color_mask[y0 : y0 + bh, x0 : x0 + bw]
        roi_color[comp.roi_mask > 0] = col

        if comp.case_label.startswith("isolated"):
            counts["isolated"] += 1
        elif comp.case_label == "touching_nuts":
            counts["touching_nuts"] += 1
        elif comp.case_label == "touching_screws":
            counts["touching_screws"] += 1

    overlay = cv2.addWeighted(img_bgr, 0.58, color_mask, 0.42, 0)

    # Overlay skeletons (yellow) and bounding boxes for touching clusters
    for comp in components:
        if comp.case_label == "noise":
            continue
        x0, y0, bw, bh = comp.bbox
        if comp.roi_skeleton.size > 0:
            skel_dil = cv2.dilate(comp.roi_skeleton, np.ones((max(1, thick), max(1, thick)), np.uint8))
            roi_ov = overlay[y0 : y0 + bh, x0 : x0 + bw]
            roi_ov[skel_dil > 0] = (0, 255, 255)
        if comp.case_label in ("touching_nuts", "touching_screws"):
            col = color_map[comp.case_label]
            cv2.rectangle(overlay, (x0, y0), (x0 + bw, y0 + bh), col, thick)

    title = f"Scene: {scene_type.upper()} | Regions: {sum(counts.values())}"
    sub = (
        f"Green=Isolated({counts['isolated']}) "
        f"Blue=TouchNuts({counts['touching_nuts']}) "
        f"Red=TouchScrews({counts['touching_screws']})"
    )
    return draw_corner_badge(overlay, title, sub)


def render_instances_overlay(
    img_bgr: np.ndarray,
    instances: list[InstanceMask],
    title_prefix: str = "Count",
    subtitle: str | None = None,
    debug_overlays: list[dict] | None = None,
) -> np.ndarray:
    """Render semi-transparent per-instance colored masks, crisp contours, labels, and corner badge."""
    h, w = img_bgr.shape[:2]
    scale = max(h, w) / 1500.0
    line_thick = max(1, int(round(1.8 * scale)))
    font_scale = max(0.38, 0.52 * scale)
    if len(instances) > 120:
        font_scale *= 0.62
    elif len(instances) > 50:
        font_scale *= 0.80
    font_thick = max(1, int(round(1.5 * scale)))

    color_mask = np.zeros_like(img_bgr)
    rng = np.random.default_rng(seed=42)
    colors: list[tuple[int, int, int]] = []

    for inst in instances:
        col = tuple(int(c) for c in rng.integers(60, 255, size=3))
        colors.append(col)
        x0, y0, bw, bh = inst.bbox
        roi_cm = color_mask[y0 : y0 + bh, x0 : x0 + bw]
        roi_cm[inst.mask > 0] = col

    overlay = cv2.addWeighted(img_bgr, 0.58, color_mask, 0.42, 0)

    # Draw per-instance contours (preserving internal holes via RETR_CCOMP)
    for inst in instances:
        x0, y0, _, _ = inst.bbox
        cnts, _ = cv2.findContours(inst.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        offset = np.array([[[x0, y0]]], dtype=np.int32)
        shifted = [c + offset for c in cnts]
        cv2.drawContours(overlay, shifted, -1, (0, 0, 255), line_thick, cv2.LINE_AA)

    # Optional technique-specific geometric debug annotations
    if debug_overlays:
        dbg_thick = max(1, int(round(2.0 * scale)))
        for dbg in debug_overlays:
            # Nut validated holes
            for h_info in dbg.get("holes", []):
                cx, cy = h_info["center_global"]
                r = max(2, int(round(h_info["radius"])))
                cv2.circle(overlay, (cx, cy), r, (0, 255, 0), dbg_thick, cv2.LINE_AA)
            # Skeleton axes, heads, tips
            for axis in dbg.get("axes", []):
                if len(axis) >= 2:
                    cv2.polylines(overlay, [axis.reshape(-1, 1, 2)], False, (0, 255, 255), dbg_thick, cv2.LINE_AA)
            for hx, hy in dbg.get("heads", []):
                cv2.circle(overlay, (hx, hy), max(3, int(round(5 * scale))), (0, 200, 255), -1)
            for tx, ty in dbg.get("tips", []):
                cv2.circle(overlay, (tx, ty), max(2, int(round(3 * scale))), (255, 255, 0), -1)
            # Hough lines
            for p1, p2 in dbg.get("lines", []):
                cv2.line(overlay, p1, p2, (255, 255, 0), dbg_thick, cv2.LINE_AA)
            # Template matched bounding boxes
            for m in dbg.get("matches", []):
                bx0, by0, bx1, by1 = m["box"]
                cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0, 255, 120), dbg_thick)
            # Concavity cut lines
            for p1, p2 in dbg.get("cuts", []):
                cv2.line(overlay, p1, p2, (0, 255, 255), max(2, dbg_thick + 1), cv2.LINE_AA)
            # Constrained watershed validated peaks
            for px, py in dbg.get("peaks", []):
                cv2.circle(overlay, (px, py), max(3, int(round(5 * scale))), (255, 0, 255), -1)

    # Draw centroid index numbers
    for idx, inst in enumerate(instances, start=1):
        cx, cy = inst.center
        cv2.putText(
            overlay,
            f"#{idx}",
            (cx - int(8 * scale), cy + int(4 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA,
        )

    return draw_corner_badge(overlay, f"{title_prefix}: {len(instances)}", subtitle)


def save_isolated_masks(
    img_shape: tuple[int, int],
    instances: list[InstanceMask],
    label_mask_path: Path,
    isolated_dir: Path,
) -> None:
    """
    Save both:
      1. A full-resolution 16-bit instance label mask PNG (0=bg, 1..N=instance IDs)
      2. Individual cropped isolated binary mask PNGs per instance in isolated_dir/
    """
    h, w = img_shape
    label_map = np.zeros((h, w), dtype=np.uint16)
    isolated_dir.mkdir(parents=True, exist_ok=True)

    for idx, inst in enumerate(instances, start=1):
        x0, y0, bw, bh = inst.bbox
        roi_lbl = label_map[y0 : y0 + bh, x0 : x0 + bw]
        roi_lbl[inst.mask > 0] = idx
        save_png(isolated_dir / f"instance_{idx:03d}.png", inst.mask)

    save_png(label_mask_path, label_map)
