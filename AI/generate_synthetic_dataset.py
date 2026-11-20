from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

AI_DIR = Path(__file__).resolve().parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from sam_segment import (  # noqa: E402
    build_sam2_predictor,
    compute_contrast_map,
    refine_mask_holes,
    run_tiled_sam2,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Group source photos into coherent physical fastener families
SOURCE_FAMILIES: dict[str, tuple[str, list[str]]] = {
    "pan_head": ("screws", ["20240713_192951", "20240713_193135"]),
    "black_flat": ("screws", ["20240713_193650", "20240713_193659"]),
    "dense_flat": ("screws", ["20240713_193831", "20240713_193839", "20240713_193907"]),
    "small_mixed": (
        "screws",
        [
            "20240713_194200",
            "20240713_194206",
            "20240713_194215",
            "20240713_194232",
            "20240713_194256",
            "20240713_194316",
        ],
    ),
    "brass_flat": (
        "screws",
        ["img1", "img1_43_nosy", "img2", "img3", "img4", "img5", "img6"],
    ),
    "hex_nut": (
        "nuts",
        [
            "20240713_194541",
            "20240713_194551",
            "20240713_194606",
            "20240713_194621",
            "20240713_194630",
        ],
    ),
}

SCREW_SUBTYPES = ["pan_head", "black_flat", "dense_flat", "small_mixed", "brass_flat"]
DENSITIES = ["sparse", "medium", "dense"]
BACKGROUNDS = ["gray_surface", "uneven_lighting", "vignetting"]

# Target major dimension range (pixels) on a 640x640 canvas
SUBTYPE_TARGET_DIM = {
    "pan_head": (72.0, 126.0),
    "black_flat": (74.0, 128.0),
    "dense_flat": (52.0, 92.0),
    "small_mixed": (48.0, 86.0),
    "brass_flat": (50.0, 90.0),
    "hex_nut": (34.0, 56.0),
}

# Worker global asset cache (populated via initializer in ProcessPoolExecutor)
_WORKER_ASSETS: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}


def _init_worker(asset_paths_by_subtype: dict[str, list[str]]) -> None:
    """Load BGRA asset arrays once per worker process for fast composition."""
    global _WORKER_ASSETS
    _WORKER_ASSETS = {}
    for subtype, paths in asset_paths_by_subtype.items():
        loaded: list[tuple[np.ndarray, np.ndarray]] = []
        for p_str in paths:
            bgra = cv2.imread(p_str, cv2.IMREAD_UNCHANGED)
            if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
                continue
            loaded.append((bgra[:, :, :3].copy(), bgra[:, :, 3].copy()))
        _WORKER_ASSETS[subtype] = loaded


def _estimate_bgr_background(img_bgr: np.ndarray) -> np.ndarray:
    """Estimate smooth 3-channel background illumination to gain-compensate vignetted cutouts."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 800.0 / max(h, w))
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)

    ksize = max(25, int(min(sh, sw) * 0.085) | 1)
    pad = ksize // 2 + 1
    small_pad = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg_pad = cv2.morphologyEx(small_pad, cv2.MORPH_CLOSE, kernel)
    blur_k = max(15, (ksize // 2) | 1)
    bg_pad = cv2.GaussianBlur(bg_pad, (blur_k, blur_k), 0)
    bg_small = bg_pad[pad : pad + sh, pad : pad + sw]
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)


def _has_interior_hole(alpha: np.ndarray) -> bool:
    """Verify a nut cutout mask contains a valid transparent interior hole."""
    cnts, hierarchy = cv2.findContours(alpha, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or hierarchy is None:
        return False
    hier = hierarchy[0]
    outer_areas = [cv2.contourArea(cnts[i]) for i, h in enumerate(hier) if h[3] == -1]
    if not outer_areas:
        return False
    max_outer = max(outer_areas)
    for i, h in enumerate(hier):
        if h[3] != -1 and cv2.contourArea(cnts[i]) >= 0.05 * max_outer:
            return True
    return False


def _save_asset_contact_sheet(assets_dir: Path, saved_by_subtype: dict[str, list[Path]]) -> None:
    """Render a quick visual contact sheet showing extracted SAM 2.1 cutouts per subtype."""
    cols = 10
    cell = 110
    subtypes = list(SOURCE_FAMILIES.keys())
    canvas = np.full((len(subtypes) * cell, cols * cell + 180, 3), 210, dtype=np.uint8)

    for r_idx, subtype in enumerate(subtypes):
        y0 = r_idx * cell
        cv2.putText(
            canvas,
            f"{subtype} ({len(saved_by_subtype[subtype])})",
            (10, y0 + cell // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        paths = saved_by_subtype[subtype]
        if not paths:
            continue
        step = max(1, len(paths) // cols)
        samples = paths[::step][:cols]
        for c_idx, p in enumerate(samples):
            bgra = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if bgra is None:
                continue
            ah, aw = bgra.shape[:2]
            scale = min((cell - 16) / max(1, aw), (cell - 16) / max(1, ah))
            nw, nh = max(1, int(round(aw * scale))), max(1, int(round(ah * scale)))
            rs = cv2.resize(bgra, (nw, nh), interpolation=cv2.INTER_AREA)
            x0 = 180 + c_idx * cell + (cell - nw) // 2
            yc = y0 + (cell - nh) // 2
            alpha_f = rs[:, :, 3:4].astype(np.float32) / 255.0
            roi = canvas[yc : yc + nh, x0 : x0 + nw].astype(np.float32)
            canvas[yc : yc + nh, x0 : x0 + nw] = np.clip(
                rs[:, :, :3].astype(np.float32) * alpha_f + roi * (1.0 - alpha_f), 0, 255
            ).astype(np.uint8)

    cv2.imwrite(str(assets_dir / "asset_samples.png"), canvas)


def extract_assets_with_sam2(
    images_dir: Path,
    assets_dir: Path,
    weights_dir: Path,
    reuse_masks_dir: Path | None,
    fresh_sam: bool = False,
) -> dict[str, list[Path]]:
    """
    Stage 1: Extract pristine isolated screw & nut BGRA assets from source images using SAM 2.1.
    Rejects touching objects, border-clipped objects, and split fragments.
    """
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    stem_to_family: dict[str, tuple[str, str]] = {}
    for subtype, (part_type, stems) in SOURCE_FAMILIES.items():
        (assets_dir / part_type / subtype).mkdir(parents=True, exist_ok=True)
        for st in stems:
            stem_to_family[st] = (part_type, subtype)

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.stem in stem_to_family
    )

    predictor = None
    saved_by_subtype: dict[str, list[Path]] = {k: [] for k in SOURCE_FAMILIES}
    clear_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    print(f"Stage 1: Extracting isolated SAM 2.1 assets from {len(image_paths)} source images...")

    for img_path in image_paths:
        stem = img_path.stem
        part_type, subtype = stem_to_family[stem]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        h, w = img_bgr.shape[:2]

        cached_mask_path = reuse_masks_dir / f"{stem}_labels.png" if reuse_masks_dir else None
        if not fresh_sam and cached_mask_path is not None and cached_mask_path.exists():
            label_map = cv2.imread(str(cached_mask_path), cv2.IMREAD_UNCHANGED)
            if label_map.ndim == 3:
                label_map = label_map[:, :, 0]
            source_mode = "cached SAM 2.1 label mask"
        else:
            if predictor is None:
                predictor = build_sam2_predictor(weights_dir, "sam2.1_b.pt")
            contrast_map = compute_contrast_map(img_bgr)
            dets = run_tiled_sam2(predictor, img_bgr, contrast_map)
            dets = refine_mask_holes(dets, contrast_map)
            label_map = np.zeros((h, w), dtype=np.uint16)
            for idx, det in enumerate(dets, start=1):
                x0, y0, x1, y1 = det["bbox"]
                label_map[y0:y1, x0:x1][det["mask"] > 0] = idx
            source_mode = "live SAM 2.1 inference"

        # Gain-compensate illumination & corner vignetting on source photo before cutting assets
        bg_bgr = _estimate_bgr_background(img_bgr).astype(np.float32)
        img_comp = np.clip(img_bgr.astype(np.float32) * (195.0 / np.maximum(bg_bgr, 1.0)), 0, 255).astype(np.uint8)

        counts = np.bincount(label_map.ravel())
        valid_ids = [int(lbl) for lbl in np.nonzero(counts)[0] if lbl > 0]
        if not valid_ids:
            continue

        areas = np.array([int(counts[lbl]) for lbl in valid_ids], dtype=np.float32)
        med_area = float(np.median(areas))
        min_area = 0.55 * med_area
        max_area = 1.55 * med_area

        # Tight bounding boxes for all instances via findContours on individual masks or slicing
        extracted_here = 0
        for lbl in valid_ids:
            area = int(counts[lbl])
            if area < min_area or area > max_area:
                continue

            ys, xs = np.where(label_map == lbl)
            if ys.size == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1

            # Skip outer-image-border clipped instances
            if x0 <= 4 or y0 <= 4 or x1 >= w - 4 or y1 >= h - 4:
                continue

            # Check 6px clearance around this instance against all other SAM masks
            pad = 10
            py0, py1 = max(0, y0 - pad), min(h, y1 + pad)
            px0, px1 = max(0, x0 - pad), min(w, x1 + pad)
            roi_lbl = label_map[py0:py1, px0:px1]
            inst_bin = (roi_lbl == lbl).astype(np.uint8)
            other_bin = (roi_lbl > 0) & (roi_lbl != lbl)
            dilated = cv2.dilate(inst_bin, clear_kernel)
            if np.any((dilated > 0) & other_bin):
                continue

            alpha = (roi_lbl[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0] == lbl).astype(np.uint8) * 255
            if part_type == "nuts" and not _has_interior_hole(alpha):
                continue

            crop_bgr = img_comp[y0:y1, x0:x1]
            bgra = np.dstack([crop_bgr, alpha])
            out_path = assets_dir / part_type / subtype / f"{stem}_id{lbl:03d}.png"
            cv2.imwrite(str(out_path), bgra)
            saved_by_subtype[subtype].append(out_path)
            extracted_here += 1

        print(f"  [{img_path.name:20s}] ({subtype:11s}, {source_mode}) -> extracted {extracted_here:3d} isolated assets")

    for subtype, paths in saved_by_subtype.items():
        if not paths:
            raise RuntimeError(f"No isolated assets extracted for subtype '{subtype}'!")

    metadata = {
        subtype: {
            "part_type": SOURCE_FAMILIES[subtype][0],
            "count": len(paths),
            "files": [str(p) for p in paths],
        }
        for subtype, paths in saved_by_subtype.items()
    }
    (assets_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    _save_asset_contact_sheet(assets_dir, saved_by_subtype)
    total_assets = sum(len(v) for v in saved_by_subtype.values())
    print(f"Extracted {total_assets} pristine isolated assets total -> {assets_dir}\n")
    return saved_by_subtype


def generate_background(h: int, w: int, bg_type: str, rng: np.random.Generator) -> np.ndarray:
    """
    Stage 2: Generate realistic surface background (gray_surface, uneven_lighting, or vignetting).
    """
    base_luma = float(rng.uniform(156.0, 220.0))
    tint = rng.uniform(-8.0, 9.0, size=3).astype(np.float32)  # BGR slight warm/cool cast
    base = np.empty((h, w, 3), dtype=np.float32)
    base[:, :, 0] = base_luma + tint[0]
    base[:, :, 1] = base_luma + tint[1]
    base[:, :, 2] = base_luma + tint[2]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    if bg_type == "gray_surface":
        # Subtle low-frequency table texture / occasional faint smudge
        if rng.random() < 0.45:
            sx = float(rng.uniform(0.15 * w, 0.85 * w))
            sy = float(rng.uniform(0.15 * h, 0.85 * h))
            sig = float(rng.uniform(18.0, 48.0))
            smudge = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * sig * sig))
            base -= smudge[:, :, None] * float(rng.uniform(4.0, 11.0))

    elif bg_type == "uneven_lighting":
        # Smooth directional linear ramp + off-center 2D Gaussian illumination gradient
        nx = (xx / max(1.0, w - 1)) - 0.5
        ny = (yy / max(1.0, h - 1)) - 0.5
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        ramp = nx * np.cos(theta) + ny * np.sin(theta)
        ramp_gain = 1.0 + ramp * float(rng.uniform(0.16, 0.34))

        gx = float(rng.uniform(0.10 * w, 0.90 * w))
        gy = float(rng.uniform(0.10 * h, 0.90 * h))
        gsig = float(rng.uniform(0.32 * max(h, w), 0.65 * max(h, w)))
        spot = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * gsig * gsig))
        spot_amp = float(rng.uniform(-0.14, 0.22))
        illum = np.clip(ramp_gain + spot * spot_amp, 0.55, 1.25)
        base *= illum[:, :, None]

    elif bg_type == "vignetting":
        # Radial optical falloff with slightly jittered center
        cx = w * (0.5 + float(rng.uniform(-0.08, 0.08)))
        cy = h * (0.5 + float(rng.uniform(-0.08, 0.08)))
        r = np.hypot(xx - cx, yy - cy)
        r_max = float(np.hypot(max(cx, w - cx), max(cy, h - cy)))
        k = float(rng.uniform(0.32, 0.65))
        p = float(rng.uniform(1.8, 3.0))
        falloff = np.clip(1.0 - k * ((r / max(1.0, r_max)) ** p), 0.25, 1.0)
        base *= falloff[:, :, None]
    else:
        raise ValueError(f"Unknown background type: {bg_type}")

    # Subtle realistic camera sensor noise
    noise_sigma = float(rng.uniform(1.6, 4.2))
    noise = rng.normal(0.0, noise_sigma, size=(h, w, 3)).astype(np.float32)
    return np.clip(base + noise, 8, 250).astype(np.uint8)


def _transform_cutout(
    bgr: np.ndarray,
    alpha: np.ndarray,
    target_major: float,
    angle_deg: float,
    bright_mul: float,
    contrast_mul: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Scale, rotate (without black edge bleed), and tightly crop a cutout."""
    h, w = bgr.shape[:2]
    scale = target_major / max(1.0, float(max(h, w)))
    nw = max(4, int(round(w * scale)))
    nh = max(4, int(round(h * scale)))

    interp_img = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    bgr_s = cv2.resize(bgr, (nw, nh), interpolation=interp_img)
    alpha_s = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Pad BGR with BORDER_REPLICATE so rotation never blends black (0,0,0) into the object rim
    diag = int(np.ceil(np.hypot(nw, nh))) + 6
    ox = (diag - nw) // 2
    oy = (diag - nh) // 2

    pad_bgr = cv2.copyMakeBorder(bgr_s, oy, diag - nh - oy, ox, diag - nw - ox, cv2.BORDER_REPLICATE)
    pad_alpha = np.zeros((diag, diag), dtype=np.uint8)
    pad_alpha[oy : oy + nh, ox : ox + nw] = alpha_s

    mat = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), angle_deg, 1.0)
    rot_bgr = cv2.warpAffine(pad_bgr, mat, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rot_alpha = cv2.warpAffine(pad_alpha, mat, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=0)

    ys, xs = np.where(rot_alpha >= 128)
    if ys.size < 25:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    crop_bgr = rot_bgr[y0:y1, x0:x1].astype(np.float32)
    crop_alpha = (rot_alpha[y0:y1, x0:x1] >= 128).astype(np.uint8) * 255

    # Subtle per-object brightness / contrast jitter
    mean_val = float(crop_bgr[crop_alpha > 0].mean())
    crop_bgr = np.clip((crop_bgr - mean_val) * contrast_mul + mean_val * bright_mul, 0, 255).astype(np.uint8)
    return crop_bgr, crop_alpha


def merge_multi_segment(segments: list[np.ndarray]) -> np.ndarray:
    """
    Connect disjoint visible polygon components of a single occluded object via
    closest-point zero-width bridges so YOLO-seg encodes the full visible modal mask.
    """
    if len(segments) == 1:
        return segments[0]

    # Greedy nearest-component chaining starting from the largest visible fragment
    ordered = sorted(segments, key=lambda s: cv2.contourArea(s), reverse=True)
    merged = ordered[0].reshape(-1, 2)

    for nxt in ordered[1:]:
        nxt_pts = nxt.reshape(-1, 2)
        diff = merged[:, None, :] - nxt_pts[None, :, :]
        dist2 = np.sum(diff.astype(np.float64) ** 2, axis=2)
        idx1, idx2 = np.unravel_index(int(np.argmin(dist2)), dist2.shape)
        # Splice nxt_pts loop cleanly at (idx1 <-> idx2) and return along the bridge
        reordered_nxt = np.roll(nxt_pts, -idx2, axis=0)
        bridge_loop = np.vstack(
            [
                merged[: idx1 + 1],
                reordered_nxt,
                reordered_nxt[:1],
                merged[idx1:],
            ]
        )
        merged = bridge_loop

    return merged


def modal_mask_to_yolo_line(vis_mask: np.ndarray, img_h: int, img_w: int) -> str | None:
    """Convert a visible binary modal mask into a normalized YOLO segmentation polygon line."""
    cnts, _ = cv2.findContours(vis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    polys: list[np.ndarray] = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < 20.0:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 1e-6:
            continue
        approx = cv2.approxPolyDP(cnt, max(1.0, 0.004 * perim), True)
        if len(approx) < 3:
            approx = cnt
        if len(approx) < 3:
            continue
        polys.append(approx.reshape(-1, 2))

    if not polys:
        return None

    merged = merge_multi_segment(polys)
    if len(merged) < 3:
        return None

    pts = merged.astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0] / float(img_w), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(img_h), 0.0, 1.0)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
    return f"0 {coords}"


def _draw_yolo_label_preview(
    img_bgr: np.ndarray,
    label_lines: list[str],
    header_text: str,
) -> np.ndarray:
    """Render side-by-side preview: Left=Synthetic RGB, Right=YOLO Modal Polygons + BBoxes."""
    h, w = img_bgr.shape[:2]
    color_mask = np.zeros_like(img_bgr)
    rng = np.random.default_rng(seed=42)

    parsed_polys: list[np.ndarray] = []
    for line in label_lines:
        vals = [float(v) for v in line.split()[1:]]
        pts = np.array(vals, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= float(w)
        pts[:, 1] *= float(h)
        poly = np.round(pts).astype(np.int32)
        parsed_polys.append(poly)
        col = tuple(int(c) for c in rng.integers(60, 255, size=3))
        cv2.fillPoly(color_mask, [poly], col)

    right = cv2.addWeighted(img_bgr, 0.52, color_mask, 0.48, 0)
    for idx, poly in enumerate(parsed_polys, start=1):
        cv2.polylines(right, [poly], True, (0, 0, 255), 1, cv2.LINE_AA)
        x0, y0 = int(poly[:, 0].min()), int(poly[:, 1].min())
        x1, y1 = int(poly[:, 0].max()), int(poly[:, 1].max())
        cv2.rectangle(right, (x0, y0), (x1, y1), (0, 255, 180), 1)
        cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
        cv2.putText(
            right,
            f"{idx}",
            (cx - 5, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(right, (8, 8), (min(w - 8, 520), 42), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        right,
        header_text,
        (16, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.hstack([img_bgr, right])


def synthesize_one_image(task: dict) -> dict:
    """Synthesize one image with balanced arrangement, background, and visible modal occlusion masks."""
    idx: int = task["idx"]
    split: str = task["split"]
    part_type: str = task["part_type"]
    subtype: str = task["subtype"]
    density: str = task["density"]
    bg_type: str = task["bg_type"]
    img_size: int = task["img_size"]
    seed: int = task["seed"]
    out_dir = Path(task["out_dir"])
    save_preview: bool = task["save_preview"]

    rng = np.random.default_rng(seed)
    h = w = img_size

    canvas = generate_background(h, w, bg_type, rng)
    canvas_f = canvas.astype(np.float32)
    modal_label_map = np.zeros((h, w), dtype=np.uint16)

    cutouts = _WORKER_ASSETS[subtype]
    lo_dim, hi_dim = SUBTYPE_TARGET_DIM[subtype]
    scene_base_dim = float(rng.uniform(lo_dim, hi_dim))

    if density == "sparse":
        target_n = int(rng.integers(5, 17)) if part_type == "screws" else int(rng.integers(6, 21))
        max_overlap = 0.04
        cluster_centers = []
    elif density == "medium":
        target_n = int(rng.integers(18, 43)) if part_type == "screws" else int(rng.integers(24, 59))
        max_overlap = 0.30
        n_cl = int(rng.integers(1, 4))
        cluster_centers = [
            (float(rng.uniform(0.24 * w, 0.76 * w)), float(rng.uniform(0.24 * h, 0.76 * h)))
            for _ in range(n_cl)
        ]
    else:  # dense
        target_n = int(rng.integers(42, 89)) if part_type == "screws" else int(rng.integers(58, 116))
        max_overlap = 0.50
        cluster_centers = [
            (float(rng.uniform(0.42 * w, 0.58 * w)), float(rng.uniform(0.42 * h, 0.58 * h)))
        ]

    # Single consistent overhead directional shadow per scene
    sh_dx = int(rng.integers(-4, 5))
    sh_dy = int(rng.integers(-4, 5))
    sh_darkness = float(rng.uniform(0.14, 0.24))

    orig_areas: dict[int, int] = {}
    current_vis: dict[int, int] = {}
    placed_count = 0

    for _ in range(target_n):
        src_bgr, src_alpha = cutouts[int(rng.integers(0, len(cutouts)))]
        obj_dim = scene_base_dim * float(rng.uniform(0.94, 1.06))
        angle = float(rng.uniform(0.0, 360.0))
        bright_mul = float(rng.uniform(0.88, 1.12))
        contrast_mul = float(rng.uniform(0.90, 1.10))

        transformed = _transform_cutout(src_bgr, src_alpha, obj_dim, angle, bright_mul, contrast_mul)
        if transformed is None:
            continue
        obj_bgr, obj_alpha = transformed
        oh, ow = obj_alpha.shape
        if oh >= h - 12 or ow >= w - 12:
            continue

        obj_mask = obj_alpha > 0
        obj_area = int(np.count_nonzero(obj_mask))
        if obj_area < 40:
            continue

        half_w, half_h = ow // 2, oh // 2
        min_cx, max_cx = half_w + 4, w - (ow - half_w) - 4
        min_cy, max_cy = half_h + 4, h - (oh - half_h) - 4
        if max_cx <= min_cx or max_cy <= min_cy:
            continue

        accepted = False
        best_pos = None

        for _attempt in range(36):
            if density == "sparse":
                cx = int(rng.integers(min_cx, max_cx + 1))
                cy = int(rng.integers(min_cy, max_cy + 1))
            elif density == "medium":
                if rng.random() < 0.55 and cluster_centers:
                    ccx, ccy = cluster_centers[int(rng.integers(0, len(cluster_centers)))]
                    cx = int(np.clip(round(rng.normal(ccx, 0.16 * w)), min_cx, max_cx))
                    cy = int(np.clip(round(rng.normal(ccy, 0.16 * h)), min_cy, max_cy))
                else:
                    cx = int(rng.integers(min_cx, max_cx + 1))
                    cy = int(rng.integers(min_cy, max_cy + 1))
            else:  # dense
                ccx, ccy = cluster_centers[0]
                sig = 0.21 * w if part_type == "screws" else 0.18 * w
                cx = int(np.clip(round(rng.normal(ccx, sig)), min_cx, max_cx))
                cy = int(np.clip(round(rng.normal(ccy, sig)), min_cy, max_cy))

            x0 = cx - half_w
            y0 = cy - half_h
            x1 = x0 + ow
            y1 = y0 + oh

            roi_lbl = modal_label_map[y0:y1, x0:x1]
            covered_vals = roi_lbl[obj_mask]
            n_covered = int(np.count_nonzero(covered_vals))
            if (n_covered / float(obj_area)) > max_overlap:
                continue

            # Ensure new object does NOT bury any existing lower object below 35% visibility
            if n_covered > 0:
                pos_cov = covered_vals[covered_vals > 0]
                binc = np.bincount(pos_cov)
                bury_ok = True
                for prev_id in np.nonzero(binc)[0]:
                    rem = current_vis[int(prev_id)] - int(binc[prev_id])
                    if rem < 0.35 * orig_areas[int(prev_id)]:
                        bury_ok = False
                        break
                if not bury_ok:
                    continue
            else:
                binc = None

            best_pos = (x0, y0, x1, y1, binc)
            accepted = True
            break

        if not accepted or best_pos is None:
            continue

        x0, y0, x1, y1, binc = best_pos
        placed_count += 1
        inst_id = placed_count
        orig_areas[inst_id] = obj_area
        current_vis[inst_id] = obj_area
        if binc is not None:
            for prev_id in np.nonzero(binc)[0]:
                current_vis[int(prev_id)] -= int(binc[prev_id])

        # Soft directional drop shadow behind placed object
        sx0 = max(0, x0 + sh_dx)
        sy0 = max(0, y0 + sh_dy)
        sx1 = min(w, x1 + sh_dx)
        sy1 = min(h, y1 + sh_dy)
        if sx1 > sx0 and sy1 > sy0:
            ax0 = sx0 - (x0 + sh_dx)
            ay0 = sy0 - (y0 + sh_dy)
            ax1 = ax0 + (sx1 - sx0)
            ay1 = ay0 + (sy1 - sy0)
            sh_mask = obj_alpha[ay0:ay1, ax0:ax1].astype(np.float32) / 255.0
            sh_blur = cv2.GaussianBlur(sh_mask, (7, 7), 0) * sh_darkness
            canvas_f[sy0:sy1, sx0:sx1] *= (1.0 - sh_blur[:, :, None])

        # Smooth 1.5px interior alpha feather (avoids hard halos while preserving nut holes)
        dist_in = cv2.distanceTransform(obj_mask.astype(np.uint8), cv2.DIST_L2, 3)
        alpha_f = np.clip(dist_in / 1.5, 0.0, 1.0)[:, :, None]
        roi_rgb = canvas_f[y0:y1, x0:x1]
        canvas_f[y0:y1, x0:x1] = obj_bgr.astype(np.float32) * alpha_f + roi_rgb * (1.0 - alpha_f)

        # Painter's algorithm Z-buffer overwrite: top object claims modal pixels cleanly
        modal_label_map[y0:y1, x0:x1][obj_mask] = inst_id

    img_final = np.clip(canvas_f, 0, 255).astype(np.uint8)

    # Export pixel-exact visible modal YOLO-seg polygon labels
    label_lines: list[str] = []
    vis_counts = np.bincount(modal_label_map.ravel())
    for inst_id in range(1, placed_count + 1):
        vis_area = int(vis_counts[inst_id]) if inst_id < len(vis_counts) else 0
        if vis_area < max(40, int(0.30 * orig_areas[inst_id])):
            continue
        vis_mask = (modal_label_map == inst_id).astype(np.uint8) * 255
        line = modal_mask_to_yolo_line(vis_mask, h, w)
        if line is not None:
            label_lines.append(line)

    stem = f"syn_{idx + 1:06d}"
    img_out = out_dir / "images" / split / f"{stem}.jpg"
    lbl_out = out_dir / "labels" / split / f"{stem}.txt"
    cv2.imwrite(str(img_out), img_final, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lbl_out.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))

    if save_preview:
        header = f"{subtype} | {density} | {bg_type} | Count: {len(label_lines)}"
        prev_img = _draw_yolo_label_preview(img_final, label_lines, header)
        prev_out = out_dir / "previews" / f"{part_type}_{density}_{bg_type}_{stem}.png"
        cv2.imwrite(str(prev_out), prev_img)

    return {
        "filename": f"{stem}.jpg",
        "split": split,
        "part_type": part_type,
        "subtype": subtype,
        "density": density,
        "bg_type": bg_type,
        "placed_count": placed_count,
        "labeled_count": len(label_lines),
    }


def generate_dataset(
    images_dir: Path,
    assets_dir: Path,
    out_dir: Path,
    weights_dir: Path,
    reuse_masks_dir: Path | None,
    fresh_sam: bool,
    num_images: int,
    val_ratio: float,
    img_size: int,
    workers: int,
    seed: int,
) -> None:
    t0 = time.perf_counter()

    saved_assets = extract_assets_with_sam2(
        images_dir=images_dir,
        assets_dir=assets_dir,
        weights_dir=weights_dir,
        reuse_masks_dir=reuse_masks_dir,
        fresh_sam=fresh_sam,
    )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)

    # Build strictly balanced 18-cell schedule: (2 part_types x 3 densities x 3 backgrounds)
    cells: list[tuple[str, str, str]] = [
        (pt, dens, bg)
        for pt in ("screws", "nuts")
        for dens in DENSITIES
        for bg in BACKGROUNDS
    ]
    n_val = int(round(num_images * val_ratio))

    rng = np.random.default_rng(seed)
    screw_sub_counters = { (d, b): 0 for d in DENSITIES for b in BACKGROUNDS }

    schedule: list[tuple[str, str, str, str]] = []
    for i in range(num_images):
        pt, dens, bg = cells[i % len(cells)]
        if pt == "screws":
            sub_idx = screw_sub_counters[(dens, bg)]
            subtype = SCREW_SUBTYPES[sub_idx % len(SCREW_SUBTYPES)]
            screw_sub_counters[(dens, bg)] = sub_idx + 1
        else:
            subtype = "hex_nut"
        schedule.append((pt, subtype, dens, bg))

    # Shuffle deterministically while keeping exact stratified split across the 18 cells
    indices_by_cell: dict[tuple[str, str, str], list[int]] = {}
    for i, (pt, _, dens, bg) in enumerate(schedule):
        indices_by_cell.setdefault((pt, dens, bg), []).append(i)

    val_set: set[int] = set()
    val_per_cell = n_val // len(cells)
    rem_val = n_val % len(cells)
    for c_idx, cell_key in enumerate(cells):
        c_idxs = indices_by_cell[cell_key].copy()
        rng.shuffle(c_idxs)
        take = val_per_cell + (1 if c_idx < rem_val else 0)
        val_set.update(c_idxs[:take])

    # Mark 1 preview image per (part_type, density, bg_type) cell (18 previews total)
    preview_indices: set[int] = set()
    for cell_key in cells:
        preview_indices.add(indices_by_cell[cell_key][0])

    # Shuffle overall order for clean numbering syn_000001..syn_006000
    perm = rng.permutation(num_images)
    tasks: list[dict] = []
    for out_idx, orig_i in enumerate(perm):
        pt, subtype, dens, bg = schedule[int(orig_i)]
        tasks.append(
            {
                "idx": out_idx,
                "split": "val" if int(orig_i) in val_set else "train",
                "part_type": pt,
                "subtype": subtype,
                "density": dens,
                "bg_type": bg,
                "img_size": img_size,
                "seed": int(seed * 100003 + out_idx),
                "out_dir": str(out_dir),
                "save_preview": int(orig_i) in preview_indices,
            }
        )

    asset_paths_str = {k: [str(p) for p in v] for k, v in saved_assets.items()}
    print(f"Stage 2 & 3: Synthesizing {num_images} images ({img_size}x{img_size}) using {workers} workers...")

    manifest_rows: list[dict] = []
    t_gen0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(asset_paths_str,)
    ) as pool:
        for done_idx, row in enumerate(pool.map(synthesize_one_image, tasks, chunksize=25), start=1):
            manifest_rows.append(row)
            if done_idx % 1000 == 0 or done_idx == num_images:
                dt = time.perf_counter() - t_gen0
                print(f"  Generated {done_idx:4d} / {num_images} images ({dt:.1f}s)")

    manifest_rows.sort(key=lambda r: r["filename"])
    manifest_csv = out_dir / "dataset_manifest.csv"
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "object"},
    }
    yaml_path = out_dir / "data.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    # Print verification summary
    split_counts = {"train": 0, "val": 0}
    part_counts = {"screws": 0, "nuts": 0}
    dens_counts = {d: 0 for d in DENSITIES}
    bg_counts = {b: 0 for b in BACKGROUNDS}
    total_instances = 0
    for r in manifest_rows:
        split_counts[r["split"]] += 1
        part_counts[r["part_type"]] += 1
        dens_counts[r["density"]] += 1
        bg_counts[r["bg_type"]] += 1
        total_instances += int(r["labeled_count"])

    print(f"\nDone in {time.perf_counter() - t0:.2f}s! Dataset saved to: {out_dir}")
    print(f"  data.yaml:     {yaml_path}")
    print(f"  manifest.csv:  {manifest_csv}")
    print(f"  previews:      {out_dir / 'previews'} ({len(preview_indices)} images)")
    print(f"  splits:        {split_counts}")
    print(f"  parts:         {part_counts}")
    print(f"  densities:     {dens_counts}")
    print(f"  backgrounds:   {bg_counts}")
    print(f"  total_masks:   {total_instances}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 6,000-image synthetic YOLO-seg dataset with modal occlusion masks"
    )
    parser.add_argument("--num-images", type=int, default=6000)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 8))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images", type=Path, default=AI_DIR.parent / "images")
    parser.add_argument("--assets-dir", type=Path, default=AI_DIR / "synthetic_assets")
    parser.add_argument("--output", type=Path, default=AI_DIR / "synthetic_dataset")
    parser.add_argument("--weights-dir", type=Path, default=AI_DIR / "weights")
    parser.add_argument(
        "--reuse-sam-masks",
        type=Path,
        default=AI_DIR / "benchmark_results" / "sam2_1_b" / "label_masks",
        help="Optional directory of cached SAM 2.1 16-bit label masks to speed up asset extraction",
    )
    parser.add_argument(
        "--fresh-sam",
        action="store_true",
        help="Force running live SAM 2.1 inference instead of reusing cached SAM 2.1 label masks",
    )
    args = parser.parse_args()

    generate_dataset(
        images_dir=args.images,
        assets_dir=args.assets_dir,
        out_dir=args.output,
        weights_dir=args.weights_dir,
        reuse_masks_dir=args.reuse_sam_masks,
        fresh_sam=args.fresh_sam,
        num_images=args.num_images,
        val_ratio=args.val_ratio,
        img_size=args.img_size,
        workers=args.workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
