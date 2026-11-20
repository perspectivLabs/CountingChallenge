from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.models.sam.predict import SAM2Predictor
from ultralytics.utils import ops

AI_DIR = Path(__file__).resolve().parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from prepare_dataset import VAL_STEMS, generate_tiles  # noqa: E402
from yolo_segment import (  # noqa: E402
    IMAGE_EXTENSIONS,
    render_yolo_overlay,
    save_label_mask,
    save_png,
)


def compute_contrast_map(img_bgr: np.ndarray) -> np.ndarray:
    """
    Illumination-compensated contrast map via morphological closing + Gaussian
    background estimation and multiplicative background division.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, 800.0 / max(h, w))
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)

    ksize = max(25, int(min(sh, sw) * 0.085) | 1)
    pad = ksize // 2 + 1
    small_pad = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg_pad = cv2.morphologyEx(small_pad, cv2.MORPH_CLOSE, kernel)
    blur_k = max(15, (ksize // 2) | 1)
    bg_pad = cv2.GaussianBlur(bg_pad, (blur_k, blur_k), 0)
    bg_small = bg_pad[pad : pad + sh, pad : pad + sw]
    background = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

    bg_f = np.maximum(background.astype(np.float32), 1.0)
    norm_gray = np.clip((gray.astype(np.float32) / bg_f) * 255.0, 0, 255).astype(np.uint8)
    return cv2.subtract(255, norm_gray)


def build_sam2_predictor(weights_dir: Path, model_name: str) -> SAM2Predictor:
    """Initialize pretrained SAM2Predictor with weights stored inside AI/weights/."""
    weights_dir = weights_dir.resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    overrides = {
        "model": str(weights_dir / model_name),
        "task": "segment",
        "mode": "predict",
        "imgsz": 1024,
        "conf": 0.25,
        "iou": 0.70,
        "device": 0,
        "verbose": False,
    }
    prev_cwd = Path.cwd()
    try:
        os.chdir(weights_dir)
        predictor = SAM2Predictor(overrides=overrides)
        predictor.setup_model()
    finally:
        os.chdir(prev_cwd)
    return predictor


def _box_pair_iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> tuple[float, int, int, int, int]:
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0, ix0, iy0, ix1, iy1
    inter = (ix1 - ix0) * (iy1 - iy0)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(1, a1 + a2 - inter), ix0, iy0, ix1, iy1


def _mask_overlap_metrics(d1: dict, d2: dict) -> tuple[float, float, int]:
    """Return (mask_iou, inter_over_min_area, mask_inter) using tight overlapping slices."""
    b1, b2 = d1["bbox"], d2["bbox"]
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0, 0.0, 0

    s1 = d1["mask"][iy0 - b1[1] : iy1 - b1[1], ix0 - b1[0] : ix1 - b1[0]] > 0
    s2 = d2["mask"][iy0 - b2[1] : iy1 - b2[1], ix0 - b2[0] : ix1 - b2[0]] > 0
    inter = int(np.count_nonzero(s1 & s2))
    if inter == 0:
        return 0.0, 0.0, 0
    union = d1["area"] + d2["area"] - inter
    min_a = min(d1["area"], d2["area"])
    return inter / max(1, union), inter / max(1, min_a), inter


def run_tiled_sam2(
    predictor: SAM2Predictor,
    img_bgr: np.ndarray,
    contrast_map: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 256,
    grid_step: int = 16,
    conf_thres: float = 0.75,
    stability_thresh: float = 0.85,
) -> list[dict]:
    """Run zero-shot SAM 2.1 automatic mask generation on foreground-filtered tile grids."""
    h, w = img_bgr.shape[:2]
    tiles = generate_tiles(h, w, tile_size=tile_size, overlap=overlap)
    raw_dets: list[dict] = []

    for tx0, ty0, tx1, ty1 in tiles:
        tile_bgr = img_bgr[ty0:ty1, tx0:tx1]
        tile_contrast = contrast_map[ty0:ty1, tx0:tx1]
        th, tw = tile_bgr.shape[:2]

        # Sample uniform grid points and keep only points on dark metallic foreground
        xs = np.arange(grid_step // 2, tw, grid_step, dtype=np.int32)
        ys = np.arange(grid_step // 2, th, grid_step, dtype=np.int32)
        if xs.size == 0 or ys.size == 0:
            continue
        gx, gy = np.meshgrid(xs, ys)
        gx_flat = gx.ravel()
        gy_flat = gy.ravel()
        fg_keep = tile_contrast[gy_flat, gx_flat] >= 24
        if not np.any(fg_keep):
            continue

        fg_xs = gx_flat[fg_keep].astype(np.float32) + 0.5
        fg_ys = gy_flat[fg_keep].astype(np.float32) + 0.5

        predictor.setup_source(tile_bgr)
        try:
            batch = next(iter(predictor.dataset))
            predictor.batch = batch
            with torch.inference_mode():
                im = predictor.preprocess(batch[1])
                predictor.features = predictor.get_im_features(im)
                ih, iw = im.shape[2:]
                # LetterBox(center=False) scales by min(ih/th, iw/tw) anchored at top-left
                r_scale = min(ih / float(th), iw / float(tw))
                point_grid = np.column_stack(
                    [fg_xs * r_scale / float(iw), fg_ys * r_scale / float(ih)]
                ).astype(np.float32)

                pred_masks, pred_scores, _ = predictor.generate(
                    im,
                    crop_n_layers=0,
                    point_grids=[point_grid],
                    points_batch_size=128,
                    conf_thres=conf_thres,
                    stability_score_thresh=stability_thresh,
                )
        finally:
            predictor.reset_image()

        if pred_masks.shape[0] == 0:
            continue

        if (pred_masks.shape[1], pred_masks.shape[2]) != (th, tw):
            scaled = ops.scale_masks(pred_masks[None].float(), (th, tw), padding=False)[0]
            masks_np = (scaled > 0.5).cpu().numpy()
        else:
            masks_np = pred_masks.bool().cpu().numpy()

        scores_np = pred_scores.cpu().numpy()
        max_tile_area = 0.25 * th * tw

        for i in range(masks_np.shape[0]):
            m = masks_np[i]
            ys_m, xs_m = np.where(m)
            area = int(ys_m.size)
            if area < 120 or area > max_tile_area:
                continue

            # Foreground darkness check to drop table background & nut inner holes
            c_vals = tile_contrast[ys_m, xs_m]
            mean_contrast = float(c_vals.mean())
            dark_frac = float(np.mean(c_vals >= 20))
            if mean_contrast < 26.0 or dark_frac < 0.55:
                continue

            iy0, iy1 = int(ys_m.min()), int(ys_m.max()) + 1
            ix0, ix1 = int(xs_m.min()), int(xs_m.max()) + 1
            gx0, gy0, gx1, gy1 = tx0 + ix0, ty0 + iy0, tx0 + ix1, ty0 + iy1

            # Reject extreme corner-hugging lens vignetting artifacts
            touches_x = (gx0 == 0) or (gx1 == w)
            touches_y = (gy0 == 0) or (gy1 == h)
            if touches_x and touches_y:
                continue

            is_border_clipped = (
                (tx0 > 0 and ix0 <= 6)
                or (tx1 < w and ix1 >= tw - 6)
                or (ty0 > 0 and iy0 <= 6)
                or (ty1 < h and iy1 >= th - 6)
            )

            tight_m = m[iy0:iy1, ix0:ix1].astype(np.uint8) * 255
            cx = int(round(tx0 + xs_m.mean()))
            cy = int(round(ty0 + ys_m.mean()))

            raw_dets.append(
                {
                    "bbox": (gx0, gy0, gx1, gy1),
                    "mask": tight_m,
                    "area": area,
                    "score": float(scores_np[i]),
                    "center": (cx, cy),
                    "is_border_clipped": bool(is_border_clipped),
                }
            )

    return resolve_sam_masks(raw_dets)


def resolve_sam_masks(dets: list[dict]) -> list[dict]:
    """
    Deduplicate masks across overlapping tiles and resolve SAM's hierarchical
    multi-scale predictions (reject multi-object super-masks and sub-part fragments).
    """
    if not dets:
        return []

    # Pass 1: Same-scale duplicate suppression (intact interior masks beat border-clipped)
    dets.sort(key=lambda d: (not d["is_border_clipped"], d["score"]), reverse=True)
    unique: list[dict] = []

    for cand in dets:
        suppressed = False
        for k in unique:
            box_iou, ix0, iy0, ix1, iy1 = _box_pair_iou(cand["bbox"], k["bbox"])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            mask_iou, contain, _ = _mask_overlap_metrics(cand, k)
            if mask_iou >= 0.50:
                suppressed = True
                break
            if box_iou >= 0.65 and mask_iou >= 0.35:
                suppressed = True
                break
            if cand["is_border_clipped"] and not k["is_border_clipped"] and contain >= 0.60:
                suppressed = True
                break
        if not suppressed:
            unique.append(cand)

    if len(unique) <= 1:
        return unique

    # Pass 2: Multi-object super-mask vs. sub-part hierarchy resolution
    def _contained_disjoint_children(parent_idx: int) -> list[int]:
        parent = unique[parent_idx]
        children_cands: list[int] = []
        for j, ch in enumerate(unique):
            if j == parent_idx or ch["area"] > 0.78 * parent["area"]:
                continue
            _, contain, _ = _mask_overlap_metrics(parent, ch)
            if contain >= 0.70:
                children_cands.append(j)
        if len(children_cands) < 2:
            return children_cands
        # Greedy maximal non-overlapping subset sorted by score/area
        children_cands.sort(key=lambda j: (unique[j]["score"], unique[j]["area"]), reverse=True)
        disjoint: list[int] = []
        for j in children_cands:
            ok = True
            for k in disjoint:
                m_iou, m_cont, _ = _mask_overlap_metrics(unique[j], unique[k])
                if m_iou >= 0.25 or m_cont >= 0.40:
                    ok = False
                    break
            if ok:
                disjoint.append(j)
        return disjoint

    children_map = {i: _contained_disjoint_children(i) for i in range(len(unique))}
    atomic_areas = [unique[i]["area"] for i, ch in children_map.items() if len(ch) < 2]
    ref_area = (
        float(np.median(atomic_areas))
        if atomic_areas
        else float(np.median([d["area"] for d in unique]))
    )

    # Drop multi-object super-masks (largest to smallest)
    order_desc = sorted(range(len(unique)), key=lambda i: unique[i]["area"], reverse=True)
    keep_flags = [True] * len(unique)

    for i in order_desc:
        ch_idxs = [j for j in children_map[i] if keep_flags[j]]
        area_i = unique[i]["area"]
        if len(ch_idxs) >= 3:
            keep_flags[i] = False
        elif len(ch_idxs) == 2:
            both_substantial = all(unique[j]["area"] >= 0.45 * ref_area for j in ch_idxs)
            # Only drop A as a 2-object super-mask when it is substantially larger than 1 object
            if area_i >= 1.48 * ref_area and both_substantial:
                keep_flags[i] = False

    # Drop sub-part fragments contained inside a surviving larger mask
    survivors = [unique[i] for i in order_desc if keep_flags[i]]
    final_list: list[dict] = []
    for d in survivors:
        is_subpart = False
        for kept in final_list:
            if d["area"] <= kept["area"]:
                _, _, inter = _mask_overlap_metrics(d, kept)
                if inter / max(1, d["area"]) >= 0.65:
                    is_subpart = True
                    break
        if not is_subpart:
            final_list.append(d)

    # Pass 3: Tiny outlier speck removal
    if len(final_list) >= 5:
        med_area = float(np.median([d["area"] for d in final_list]))
        final_list = [d for d in final_list if d["area"] >= 0.28 * med_area]

    final_list.sort(key=lambda d: (d["center"][1], d["center"][0]))
    return final_list


def _circularity(cnt: np.ndarray) -> float:
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if perim <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (perim * perim))


def refine_mask_holes(dets: list[dict], contrast_map: np.ndarray) -> list[dict]:
    """
    Preserve interior circular bright holes on hollow ring objects (e.g., hex nuts)
    without performing any per-object class categorization.
    """
    for det in dets:
        x0, y0, x1, y1 = det["bbox"]
        mask = det["mask"]
        roi_contrast = contrast_map[y0:y1, x0:x1]

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        outer_cnt = max(cnts, key=cv2.contourArea)
        filled_area = float(cv2.contourArea(outer_cnt))
        circ = _circularity(outer_cnt)
        rect = cv2.minAreaRect(outer_cnt)
        rw, rh = float(rect[1][0]), float(rect[1][1])
        aspect_ratio = max(rw, rh) / max(min(rw, rh), 1e-6)

        if aspect_ratio <= 1.40 and circ >= 0.68 and filled_area > 0:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, [outer_cnt], -1, 255, thickness=cv2.FILLED)

            # Bright regions strictly interior to filled outer polygon
            bright_inside = ((filled > 0) & (roi_contrast < 24)).astype(np.uint8)
            outside_pad = cv2.copyMakeBorder((filled == 0).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
            outside_dil = cv2.dilate(outside_pad, np.ones((3, 3), np.uint8))[1:-1, 1:-1]

            n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats(bright_inside, connectivity=8)
            hole_masks: list[np.ndarray] = []
            for k in range(1, n_cc):
                comp_bin = cc_lbl == k
                if np.any(comp_bin & (outside_dil > 0)):
                    continue  # Touches outer boundary -> indentation, not closed hole
                h_area = int(cc_stats[k, cv2.CC_STAT_AREA])
                if h_area < 0.06 * filled_area:
                    continue
                h_cnts, _ = cv2.findContours(comp_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if h_cnts and _circularity(max(h_cnts, key=cv2.contourArea)) >= 0.55:
                    hole_masks.append(comp_bin)

            if hole_masks:
                refined = mask.copy()
                for hm in hole_masks:
                    refined[hm] = 0
                det["mask"] = refined
                ys, xs = np.where(refined > 0)
                if ys.size > 0:
                    det["area"] = int(ys.size)
                    det["center"] = (int(round(x0 + xs.mean())), int(round(y0 + ys.mean())))

    return dets


def load_existing_counts(ai_cmp_csv: Path, nonai_csv: Path) -> dict[str, dict]:
    existing: dict[str, dict] = {}
    if nonai_csv.exists():
        with nonai_csv.open("r", newline="") as f:
            for row in csv.DictReader(f):
                existing.setdefault(row["image"].strip(), {})["non_ai_adaptive"] = int(row["final_adaptive"])
    if ai_cmp_csv.exists():
        with ai_cmp_csv.open("r", newline="") as f:
            for row in csv.DictReader(f):
                name = row["image"].strip()
                entry = existing.setdefault(name, {})
                ye_val = row.get("yoloe_count") or row.get("yoloe_total") or ""
                if ye_val != "":
                    entry["yoloe_zeroshot"] = int(ye_val)
    return existing


def run_sam2_pipeline(
    model_name: str,
    images_dir: Path,
    out_root: Path,
    weights_dir: Path,
    tile_size: int,
    overlap: int,
    grid_step: int,
    conf_thres: float,
    stability_thresh: float,
    only: list[str] | None = None,
) -> None:
    sam_out = out_root / "sam2_zeroshot"
    sam_out.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if only:
        only_set = set(only)
        image_paths = [p for p in image_paths if p.stem in only_set or p.name in only_set]

    predictor = build_sam2_predictor(weights_dir, model_name)
    existing = load_existing_counts(
        out_root / "comparison_counts.csv",
        AI_DIR.parent / "Non_AI" / "adaptive_results" / "summary_counts.csv",
    )

    print(
        f"Running Zero-Shot SAM 2.1 ({model_name}, grid={grid_step}, conf={conf_thres}, stab={stability_thresh}) "
        f"on {len(image_paths)} images -> {sam_out}\n"
    )

    rows: list[dict] = []
    t_total = time.perf_counter()

    for img_path in image_paths:
        t0 = time.perf_counter()
        stem = img_path.stem
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        h, w = img_bgr.shape[:2]

        contrast_map = compute_contrast_map(img_bgr)
        dets = run_tiled_sam2(
            predictor=predictor,
            img_bgr=img_bgr,
            contrast_map=contrast_map,
            tile_size=tile_size,
            overlap=overlap,
            grid_step=grid_step,
            conf_thres=conf_thres,
            stability_thresh=stability_thresh,
        )
        dets = refine_mask_holes(dets, contrast_map)

        overlay = render_yolo_overlay(img_bgr, dets, f"Zero-Shot SAM 2.1 ({model_name})")
        save_png(sam_out / "overlays" / f"{stem}.png", overlay)
        save_label_mask((h, w), dets, sam_out / "label_masks" / f"{stem}_labels.png")

        elapsed = round(time.perf_counter() - t0, 2)

        prev = existing.get(img_path.name, {})
        non_ai = prev.get("non_ai_adaptive", "")
        yoloe = prev.get("yoloe_zeroshot", "")

        row = {
            "image": img_path.name,
            "split": "val" if stem in VAL_STEMS else "train",
            "non_ai_adaptive": non_ai,
            "yoloe_zeroshot": yoloe,
            "sam2_count": len(dets),
            "sam2_sec": elapsed,
        }
        rows.append(row)

        print(
            f"[{img_path.name:20s}] NonAI={str(non_ai):>3s} | YOLOE={str(yoloe):>3s} | "
            f"SAM2={len(dets):3d} ({elapsed:.2f}s)"
        )

    if rows:
        csv_path = out_root / "sam2_comparison_counts.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDone in {time.perf_counter() - t_total:.2f}s. Saved comparison CSV to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure Zero-Shot SAM 2.1 Instance Segmentation & Counting")
    parser.add_argument("--model", type=str, default="sam2.1_b.pt")
    parser.add_argument("--images", type=Path, default=AI_DIR.parent / "images")
    parser.add_argument("--output", type=Path, default=AI_DIR / "results")
    parser.add_argument("--weights-dir", type=Path, default=AI_DIR / "weights")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--grid-step", type=int, default=14)
    parser.add_argument("--conf-thres", type=float, default=0.68)
    parser.add_argument("--stability-thresh", type=float, default=0.80)
    parser.add_argument("--only", type=str, nargs="*", default=None)
    args = parser.parse_args()

    run_sam2_pipeline(
        model_name=args.model,
        images_dir=args.images,
        out_root=args.output,
        weights_dir=args.weights_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        grid_step=args.grid_step,
        conf_thres=args.conf_thres,
        stability_thresh=args.stability_thresh,
        only=args.only,
    )


if __name__ == "__main__":
    main()
