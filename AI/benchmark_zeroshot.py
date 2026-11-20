from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from ultralytics import FastSAM, YOLO, YOLOE
from ultralytics.models.sam.predict import Predictor as SAMPredictor
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import ops

AI_DIR = Path(__file__).resolve().parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from prepare_dataset import VAL_STEMS, generate_tiles  # noqa: E402
from sam_segment import (  # noqa: E402
    build_sam2_predictor,
    compute_contrast_map,
    refine_mask_holes,
    resolve_sam_masks,
)
from yolo_segment import (  # noqa: E402
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    YOLOE_PROMPTS,
    deduplicate_detections,
    render_yolo_overlay,
    save_label_mask,
    save_png,
)

MODEL_SPECS = [
    ("sam2_1_b", "SAM 2.1-b"),
    ("fastsam_x", "FastSAM-x"),
    ("mobilesam_v2", "MobileSAMv2"),
    ("yoloe_11l_pf", "YOLOE-11L-PF"),
    ("yoloe_11l_text", "YOLOE-11L-Text"),
]

GRID_STEMS = [
    "20240713_192951",
    "20240713_193650",
    "20240713_193831",
    "20240713_194200",
    "20240713_194541",
    "img3",
]


def _extract_filtered_tile_masks(
    masks_bool: np.ndarray,
    scores: np.ndarray,
    tile_contrast: np.ndarray,
    tx0: int,
    ty0: int,
    img_h: int,
    img_w: int,
    th: int,
    tw: int,
) -> list[dict]:
    """Filter candidate binary tile masks by area, foreground contrast, and corner borders."""
    max_tile_area = 0.25 * th * tw
    dets: list[dict] = []

    for i in range(masks_bool.shape[0]):
        m = masks_bool[i]
        ys_m, xs_m = np.where(m)
        area = int(ys_m.size)
        if area < 120 or area > max_tile_area:
            continue

        c_vals = tile_contrast[ys_m, xs_m]
        mean_contrast = float(c_vals.mean())
        dark_frac = float(np.mean(c_vals >= 20))
        if mean_contrast < 26.0 or dark_frac < 0.55:
            continue

        iy0, iy1 = int(ys_m.min()), int(ys_m.max()) + 1
        ix0, ix1 = int(xs_m.min()), int(xs_m.max()) + 1
        gx0, gy0, gx1, gy1 = tx0 + ix0, ty0 + iy0, tx0 + ix1, ty0 + iy1

        touches_x = (gx0 == 0) or (gx1 == img_w)
        touches_y = (gy0 == 0) or (gy1 == img_h)
        if touches_x and touches_y:
            continue

        tx1, ty1 = tx0 + tw, ty0 + th
        is_border_clipped = (
            (tx0 > 0 and ix0 <= 6)
            or (tx1 < img_w and ix1 >= tw - 6)
            or (ty0 > 0 and iy0 <= 6)
            or (ty1 < img_h and iy1 >= th - 6)
        )

        dets.append(
            {
                "bbox": (gx0, gy0, gx1, gy1),
                "mask": m[iy0:iy1, ix0:ix1].astype(np.uint8) * 255,
                "area": area,
                "score": float(scores[i]),
                "center": (int(round(tx0 + xs_m.mean())), int(round(ty0 + ys_m.mean()))),
                "is_border_clipped": bool(is_border_clipped),
            }
        )
    return dets


def _run_sam2_on_tile(
    predictor,
    tile_bgr: np.ndarray,
    tile_contrast: np.ndarray,
    grid_step: int = 14,
    conf_thres: float = 0.68,
    stability_thresh: float = 0.80,
) -> tuple[np.ndarray, np.ndarray, float]:
    th, tw = tile_bgr.shape[:2]
    xs = np.arange(grid_step // 2, tw, grid_step, dtype=np.int32)
    ys = np.arange(grid_step // 2, th, grid_step, dtype=np.int32)
    if xs.size == 0 or ys.size == 0:
        return np.empty((0, th, tw), dtype=bool), np.empty((0,), dtype=np.float32), 0.0
    gx, gy = np.meshgrid(xs, ys)
    gx_flat, gy_flat = gx.ravel(), gy.ravel()
    fg_keep = tile_contrast[gy_flat, gx_flat] >= 24
    if not np.any(fg_keep):
        return np.empty((0, th, tw), dtype=bool), np.empty((0,), dtype=np.float32), 0.0

    fg_xs = gx_flat[fg_keep].astype(np.float32) + 0.5
    fg_ys = gy_flat[fg_keep].astype(np.float32) + 0.5

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    predictor.setup_source(tile_bgr)
    try:
        batch = next(iter(predictor.dataset))
        predictor.batch = batch
        with torch.inference_mode():
            im = predictor.preprocess(batch[1])
            predictor.features = predictor.get_im_features(im)
            ih, iw = im.shape[2:]
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
    torch.cuda.synchronize()
    infer_dt = time.perf_counter() - t0

    if pred_masks.shape[0] == 0:
        return np.empty((0, th, tw), dtype=bool), np.empty((0,), dtype=np.float32), infer_dt

    if (pred_masks.shape[1], pred_masks.shape[2]) != (th, tw):
        scaled = ops.scale_masks(pred_masks[None].float(), (th, tw), padding=False)[0]
        masks_bool = (scaled > 0.5).cpu().numpy()
    else:
        masks_bool = pred_masks.bool().cpu().numpy()
    scores_np = pred_scores.cpu().numpy()
    return masks_bool, scores_np, infer_dt


def _run_fastsam_on_tile(
    model: FastSAM,
    tile_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    th, tw = tile_bgr.shape[:2]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = model.predict(
        tile_bgr,
        device=0,
        retina_masks=True,
        imgsz=1024,
        conf=0.25,
        iou=0.70,
        max_det=300,
        verbose=False,
    )
    torch.cuda.synchronize()
    infer_dt = time.perf_counter() - t0

    if not results or results[0].masks is None or results[0].boxes is None or len(results[0].boxes) == 0:
        return np.empty((0, th, tw), dtype=bool), np.empty((0,), dtype=np.float32), infer_dt

    r = results[0]
    masks_np = r.masks.data.cpu().numpy()
    if masks_np.shape[1:] != (th, tw):
        resized = np.empty((masks_np.shape[0], th, tw), dtype=bool)
        for idx in range(masks_np.shape[0]):
            resized[idx] = cv2.resize(masks_np[idx], (tw, th), interpolation=cv2.INTER_NEAREST) > 0.5
        masks_bool = resized
    else:
        masks_bool = masks_np > 0.5
    scores_np = r.boxes.conf.cpu().numpy()
    return masks_bool, scores_np, infer_dt


def _run_mobilesamv2_on_tile(
    obj_model: YOLO,
    sam_pred: SAMPredictor,
    tile_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    th, tw = tile_bgr.shape[:2]
    max_box_area = 0.35 * th * tw

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    obj_res = obj_model.predict(
        tile_bgr,
        imgsz=1024,
        conf=0.15,
        iou=0.70,
        max_det=300,
        device=0,
        verbose=False,
    )
    if not obj_res or obj_res[0].boxes is None or len(obj_res[0].boxes) == 0:
        torch.cuda.synchronize()
        return np.empty((0, th, tw), dtype=bool), np.empty((0,), dtype=np.float32), time.perf_counter() - t0

    boxes_xyxy = obj_res[0].boxes.xyxy.cpu().numpy()
    box_confs = obj_res[0].boxes.conf.cpu().numpy()
    box_areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    keep = (box_areas >= 80.0) & (box_areas <= max_box_area)
    valid_boxes = boxes_xyxy[keep]
    valid_confs = box_confs[keep]

    if len(valid_boxes) == 0:
        torch.cuda.synchronize()
        return np.empty((0, th, tw), dtype=bool), np.empty((0,), dtype=np.float32), time.perf_counter() - t0

    sam_pred.setup_source(tile_bgr)
    try:
        batch = next(iter(sam_pred.dataset))
        sam_pred.batch = batch
        with torch.inference_mode():
            im = sam_pred.preprocess(batch[1])
            sam_pred.features = sam_pred.get_im_features(im)
            sam_pred.segment_all = False
            all_m = []
            all_s = []
            for start in range(0, len(valid_boxes), 64):
                chunk = valid_boxes[start : start + 64]
                pm, ps = sam_pred.prompt_inference(im, bboxes=chunk, multimask_output=False)
                if pm.ndim == 4:
                    pm = pm[:, 0]
                all_m.append(pm)
                all_s.append(ps.reshape(-1))
            pred_masks = torch.cat(all_m, dim=0)
            pred_scores = torch.cat(all_s, dim=0)
    finally:
        sam_pred.reset_image()

    torch.cuda.synchronize()
    infer_dt = time.perf_counter() - t0

    if pred_masks.shape[0] == 0:
        return np.empty((0, th, tw), dtype=bool), np.empty((0,), dtype=np.float32), infer_dt

    if (pred_masks.shape[1], pred_masks.shape[2]) != (th, tw):
        scaled = ops.scale_masks(pred_masks[None].float(), (th, tw), padding=False)[0]
    else:
        scaled = pred_masks.float()
    masks_bool = (scaled > sam_pred.model.mask_threshold).cpu().numpy()
    # Combine object-prompter conf and SAM mask IoU score
    scores_np = 0.5 * valid_confs + 0.5 * pred_scores.cpu().numpy()
    return masks_bool, scores_np, infer_dt


def _run_yoloe_pf_on_tile(
    model: YOLOE,
    tile_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    th, tw = tile_bgr.shape[:2]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = model.predict(
        tile_bgr,
        imgsz=640,
        conf=0.15,
        iou=0.55,
        retina_masks=True,
        max_det=300,
        device=0,
        verbose=False,
    )
    torch.cuda.synchronize()
    infer_dt = time.perf_counter() - t0

    if not results or results[0].masks is None or results[0].boxes is None or len(results[0].boxes) == 0:
        return np.empty((0, th, tw), dtype=bool), np.empty((0,), dtype=np.float32), infer_dt

    r = results[0]
    masks_np = r.masks.data.cpu().numpy()
    if masks_np.shape[1:] != (th, tw):
        resized = np.empty((masks_np.shape[0], th, tw), dtype=bool)
        for idx in range(masks_np.shape[0]):
            resized[idx] = cv2.resize(masks_np[idx], (tw, th), interpolation=cv2.INTER_NEAREST) > 0.5
        masks_bool = resized
    else:
        masks_bool = masks_np > 0.5
    scores_np = r.boxes.conf.cpu().numpy()
    return masks_bool, scores_np, infer_dt


def _run_yoloe_text_on_image(
    model: YOLOE,
    img_bgr: np.ndarray,
    tile_size: int,
    overlap: int,
) -> tuple[list[dict], float]:
    h, w = img_bgr.shape[:2]
    tiles = generate_tiles(h, w, tile_size=tile_size, overlap=overlap)
    raw_dets: list[dict] = []
    infer_sec = 0.0

    for tx0, ty0, tx1, ty1 in tiles:
        tile_img = img_bgr[ty0:ty1, tx0:tx1]
        th, tw = tile_img.shape[:2]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model.predict(
            source=tile_img,
            conf=0.18,
            iou=0.55,
            imgsz=640,
            retina_masks=True,
            max_det=300,
            device=0,
            verbose=False,
        )
        torch.cuda.synchronize()
        infer_sec += time.perf_counter() - t0

        if not results:
            continue
        r = results[0]
        if r.masks is None or r.boxes is None or len(r.boxes) == 0:
            continue

        masks_np = r.masks.data.cpu().numpy()
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes_xyxy)):
            m = masks_np[i]
            if m.shape[0] != th or m.shape[1] != tw:
                m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
            bin_m = (m > 0.5).astype(np.uint8) * 255

            bx0 = max(0, int(np.floor(boxes_xyxy[i, 0])))
            by0 = max(0, int(np.floor(boxes_xyxy[i, 1])))
            bx1 = min(tw, int(np.ceil(boxes_xyxy[i, 2])))
            by1 = min(th, int(np.ceil(boxes_xyxy[i, 3])))
            if bx1 <= bx0 or by1 <= by0:
                continue

            sub_m = bin_m[by0:by1, bx0:bx1]
            ys, xs = np.where(sub_m > 0)
            if ys.size < 60:
                continue

            iy0, iy1 = int(ys.min()), int(ys.max()) + 1
            ix0, ix1 = int(xs.min()), int(xs.max()) + 1
            tight_m = sub_m[iy0:iy1, ix0:ix1].copy()

            gx0 = tx0 + bx0 + ix0
            gy0 = ty0 + by0 + iy0
            gx1 = tx0 + bx0 + ix1
            gy1 = ty0 + by0 + iy1

            is_border_clipped = (
                (tx0 > 0 and (bx0 + ix0) <= 6)
                or (tx1 < w and (bx0 + ix1) >= tw - 6)
                or (ty0 > 0 and (by0 + iy0) <= 6)
                or (ty1 < h and (by0 + iy1) >= th - 6)
            )
            cls_id = int(clss[i])
            raw_dets.append(
                {
                    "cls_id": cls_id,
                    "cls_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                    "conf": float(confs[i]),
                    "bbox": (gx0, gy0, gx1, gy1),
                    "mask": tight_m,
                    "area": int(ys.size),
                    "center": (int(round(tx0 + bx0 + xs.mean())), int(round(ty0 + by0 + ys.mean()))),
                    "is_border_clipped": bool(is_border_clipped),
                }
            )

    return deduplicate_detections(raw_dets), infer_sec


def _load_model_for_benchmark(model_key: str, weights_dir: Path, warmup_tile: np.ndarray):
    weights_dir = weights_dir.resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    prev_cwd = Path.cwd()

    try:
        os.chdir(weights_dir)

        if model_key == "sam2_1_b":
            predictor = build_sam2_predictor(weights_dir, "sam2.1_b.pt")
            ckpt_mb = (weights_dir / "sam2.1_b.pt").stat().st_size / (1024 ** 2)
            params_m = sum(p.numel() for p in predictor.model.parameters()) / 1e6
            # Warmup pass
            _run_sam2_on_tile(predictor, warmup_tile, compute_contrast_map(warmup_tile))
            return {"predictor": predictor}, ckpt_mb, params_m

        if model_key == "fastsam_x":
            model = FastSAM(str(weights_dir / "FastSAM-x.pt"))
            _run_fastsam_on_tile(model, warmup_tile)
            ckpt_mb = (weights_dir / "FastSAM-x.pt").stat().st_size / (1024 ** 2)
            params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
            return {"model": model}, ckpt_mb, params_m

        if model_key == "mobilesam_v2":
            obj_pt = weights_dir / "ObjectAwareModel.pt"
            if not obj_pt.exists():
                hf_hub_download(
                    repo_id="RogerQi/MobileSAMV2",
                    filename="ObjectAwareModel.pt",
                    local_dir=str(weights_dir),
                )
            obj_model = YOLO(str(obj_pt), task="detect")
            # MobileSAMv2's ObjectAwareModel stripped `proto` from a YOLOv8 Segment head
            obj_model.model.model[-1].__class__ = Detect
            obj_model.model.task = "detect"
            obj_model.overrides["task"] = "detect"
            obj_model.task = "detect"
            obj_model.predictor = None
            sam_pred = SAMPredictor(
                overrides={
                    "model": str(weights_dir / "mobile_sam.pt"),
                    "task": "segment",
                    "mode": "predict",
                    "imgsz": 1024,
                    "device": 0,
                    "verbose": False,
                }
            )
            sam_pred.setup_model()
            _run_mobilesamv2_on_tile(obj_model, sam_pred, warmup_tile)
            ckpt_mb = (
                obj_pt.stat().st_size + (weights_dir / "mobile_sam.pt").stat().st_size
            ) / (1024 ** 2)
            params_m = (
                sum(p.numel() for p in obj_model.model.parameters())
                + sum(p.numel() for p in sam_pred.model.parameters())
            ) / 1e6
            return {"obj_model": obj_model, "sam_pred": sam_pred}, ckpt_mb, params_m

        if model_key == "yoloe_11l_pf":
            model = YOLOE(str(weights_dir / "yoloe-11l-seg-pf.pt"))
            _run_yoloe_pf_on_tile(model, warmup_tile)
            ckpt_mb = (weights_dir / "yoloe-11l-seg-pf.pt").stat().st_size / (1024 ** 2)
            params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
            return {"model": model}, ckpt_mb, params_m

        if model_key == "yoloe_11l_text":
            model = YOLOE(str(weights_dir / "yoloe-11l-seg.pt"))
            text_pe = model.get_text_pe(YOLOE_PROMPTS)
            model.set_classes(YOLOE_PROMPTS, text_pe)
            # Warmup pass
            model.predict(
                source=warmup_tile,
                conf=0.18,
                iou=0.55,
                imgsz=640,
                retina_masks=True,
                max_det=300,
                device=0,
                verbose=False,
            )
            ckpt_mb = (weights_dir / "yoloe-11l-seg.pt").stat().st_size / (1024 ** 2)
            params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
            return {"model": model}, ckpt_mb, params_m

        raise ValueError(f"Unknown model key: {model_key}")
    finally:
        os.chdir(prev_cwd)


def run_single_model_benchmark(
    model_key: str,
    images_dir: Path,
    out_root: Path,
    weights_dir: Path,
    tile_size: int = 1024,
    overlap: int = 256,
    only: list[str] | None = None,
) -> dict:
    label_map = dict(MODEL_SPECS)
    model_label = label_map[model_key]
    model_out = out_root / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if only:
        only_set = set(only)
        image_paths = [p for p in image_paths if p.stem in only_set or p.name in only_set]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    first_img = cv2.imread(str(image_paths[0]))
    warmup_tile = first_img[: min(tile_size, first_img.shape[0]), : min(tile_size, first_img.shape[1])]

    bundle, ckpt_mb, params_m = _load_model_for_benchmark(model_key, weights_dir, warmup_tile)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    model_vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    torch.cuda.reset_peak_memory_stats(0)

    print(
        f"\n=== [{model_label}] ckpt={ckpt_mb:.1f}MB | params={params_m:.2f}M | "
        f"model_vram={model_vram_mb:.1f}MB | images={len(image_paths)} ==="
    )

    per_image: list[dict] = []

    for img_path in image_paths:
        stem = img_path.stem
        img_bgr = cv2.imread(str(img_path))
        h, w = img_bgr.shape[:2]

        t_pipe0 = time.perf_counter()

        if model_key == "yoloe_11l_text":
            dets, infer_sec = _run_yoloe_text_on_image(bundle["model"], img_bgr, tile_size, overlap)
        else:
            contrast_map = compute_contrast_map(img_bgr)
            tiles = generate_tiles(h, w, tile_size=tile_size, overlap=overlap)
            raw_dets: list[dict] = []
            infer_sec = 0.0

            for tx0, ty0, tx1, ty1 in tiles:
                tile_bgr = img_bgr[ty0:ty1, tx0:tx1]
                tile_contrast = contrast_map[ty0:ty1, tx0:tx1]
                if not np.any(tile_contrast >= 24):
                    continue
                th, tw = tile_bgr.shape[:2]

                if model_key == "sam2_1_b":
                    masks_bool, scores_np, dt = _run_sam2_on_tile(
                        bundle["predictor"], tile_bgr, tile_contrast
                    )
                elif model_key == "fastsam_x":
                    masks_bool, scores_np, dt = _run_fastsam_on_tile(bundle["model"], tile_bgr)
                elif model_key == "mobilesam_v2":
                    masks_bool, scores_np, dt = _run_mobilesamv2_on_tile(
                        bundle["obj_model"], bundle["sam_pred"], tile_bgr
                    )
                elif model_key == "yoloe_11l_pf":
                    masks_bool, scores_np, dt = _run_yoloe_pf_on_tile(bundle["model"], tile_bgr)
                else:
                    raise ValueError(model_key)

                infer_sec += dt
                if masks_bool.shape[0] > 0:
                    raw_dets.extend(
                        _extract_filtered_tile_masks(
                            masks_bool, scores_np, tile_contrast, tx0, ty0, h, w, th, tw
                        )
                    )

            dets = resolve_sam_masks(raw_dets)
            dets = refine_mask_holes(dets, contrast_map)

        total_pipe_sec = time.perf_counter() - t_pipe0

        subtitle = f"Zero-Shot {model_label} | Infer: {infer_sec:.2f}s"
        overlay = render_yolo_overlay(img_bgr, dets, subtitle)
        save_png(model_out / "overlays" / f"{stem}.png", overlay)
        save_label_mask((h, w), dets, model_out / "label_masks" / f"{stem}_labels.png")

        per_image.append(
            {
                "image": img_path.name,
                "split": "val" if stem in VAL_STEMS else "train",
                "count": len(dets),
                "infer_sec": round(infer_sec, 4),
                "pipe_sec": round(total_pipe_sec, 4),
            }
        )
        print(
            f"  [{img_path.name:20s}] count={len(dets):3d} | "
            f"infer={infer_sec:5.2f}s | pipe={total_pipe_sec:5.2f}s"
        )

    torch.cuda.synchronize()
    peak_alloc_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    peak_res_mb = torch.cuda.max_memory_reserved(0) / (1024 ** 2)

    infer_times = [r["infer_sec"] for r in per_image]
    pipe_times = [r["pipe_sec"] for r in per_image]
    summary = {
        "model_key": model_key,
        "model_label": model_label,
        "num_images": len(per_image),
        "checkpoint_mb": round(ckpt_mb, 2),
        "params_m": round(params_m, 2),
        "model_vram_mb": round(model_vram_mb, 1),
        "peak_alloc_mb": round(peak_alloc_mb, 1),
        "peak_reserved_mb": round(peak_res_mb, 1),
        "total_infer_sec": round(float(np.sum(infer_times)), 2),
        "mean_infer_sec": round(float(np.mean(infer_times)), 3),
        "total_pipe_sec": round(float(np.sum(pipe_times)), 2),
        "mean_pipe_sec": round(float(np.mean(pipe_times)), 3),
        "per_image": per_image,
    }
    (model_out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(
        f"-> [{model_label}] peak_alloc={peak_alloc_mb:.1f}MB peak_res={peak_res_mb:.1f}MB | "
        f"total_infer={summary['total_infer_sec']:.2f}s (mean={summary['mean_infer_sec']:.3f}s)"
    )
    return summary


def build_comparison_grids(
    images_dir: Path,
    out_root: Path,
    stems: list[str],
    panel_w: int = 900,
) -> None:
    grids_dir = out_root / "comparison_grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        orig_path = images_dir / f"{stem}.jpg"
        if not orig_path.exists():
            continue
        orig = cv2.imread(str(orig_path))
        if orig is None:
            continue
        oh, ow = orig.shape[:2]
        panel_h = int(round(panel_w * oh / ow))

        panels: list[np.ndarray] = []
        entries = [("Original", orig_path)] + [
            (label, out_root / key / "overlays" / f"{stem}.png")
            for key, label in MODEL_SPECS
        ]

        for label, img_p in entries:
            im = cv2.imread(str(img_p)) if img_p.exists() else orig.copy()
            if im is None:
                im = orig.copy()
            resized = cv2.resize(im, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            # Crisp top label strip above panel
            bar = np.zeros((44, panel_w, 3), dtype=np.uint8)
            cv2.putText(
                bar,
                label,
                (16, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(np.vstack([bar, resized]))

        top_row = np.hstack(panels[:3])
        bot_row = np.hstack(panels[3:6])
        grid = np.vstack([top_row, bot_row])
        save_png(grids_dir / f"{stem}_compare.png", grid)


def aggregate_reports(images_dir: Path, out_root: Path, only: list[str] | None) -> None:
    metrics_list: list[dict] = []
    for key, _ in MODEL_SPECS:
        mp = out_root / key / "metrics.json"
        if mp.exists():
            metrics_list.append(json.loads(mp.read_text()))

    if not metrics_list:
        return

    sam_ref = next((m for m in metrics_list if m["model_key"] == "sam2_1_b"), None)
    sam_infer = sam_ref["total_infer_sec"] if sam_ref and sam_ref["total_infer_sec"] > 0 else None

    summary_rows: list[dict] = []
    for m in metrics_list:
        speedup = round(sam_infer / m["total_infer_sec"], 2) if sam_infer and m["total_infer_sec"] > 0 else ""
        summary_rows.append(
            {
                "model_key": m["model_key"],
                "model_label": m["model_label"],
                "num_images": m["num_images"],
                "checkpoint_mb": m["checkpoint_mb"],
                "params_m": m["params_m"],
                "model_vram_mb": m["model_vram_mb"],
                "peak_alloc_mb": m["peak_alloc_mb"],
                "peak_reserved_mb": m["peak_reserved_mb"],
                "total_infer_sec": m["total_infer_sec"],
                "mean_infer_sec": m["mean_infer_sec"],
                "total_pipe_sec": m["total_pipe_sec"],
                "mean_pipe_sec": m["mean_pipe_sec"],
                "speedup_vs_sam2": speedup,
            }
        )

    summary_csv = out_root / "benchmark_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Load Non_AI Adaptive reference counts if available
    nonai_csv = AI_DIR.parent / "Non_AI" / "adaptive_results" / "summary_counts.csv"
    nonai_map: dict[str, int] = {}
    if nonai_csv.exists():
        with nonai_csv.open("r", newline="") as f:
            for r in csv.DictReader(f):
                nonai_map[r["image"].strip()] = int(r["final_adaptive"])

    by_model_img: dict[str, dict[str, dict]] = {
        m["model_key"]: {r["image"]: r for r in m["per_image"]} for m in metrics_list
    }
    ordered_images = [r["image"] for r in metrics_list[0]["per_image"]]

    per_img_rows: list[dict] = []
    for img_name in ordered_images:
        row: dict[str, object] = {
            "image": img_name,
            "non_ai_adaptive": nonai_map.get(img_name, ""),
        }
        for key, _ in MODEL_SPECS:
            entry = by_model_img.get(key, {}).get(img_name)
            if entry:
                row[f"{key}_count"] = entry["count"]
                row[f"{key}_infer_sec"] = entry["infer_sec"]
            else:
                row[f"{key}_count"] = ""
                row[f"{key}_infer_sec"] = ""
        per_img_rows.append(row)

    per_img_csv = out_root / "benchmark_per_image.csv"
    with per_img_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_img_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_img_rows)

    stems = only if only else GRID_STEMS
    build_comparison_grids(images_dir, out_root, stems)

    print("\n================ ZERO-SHOT BENCHMARK SUMMARY ================")
    for sr in summary_rows:
        print(
            f"{sr['model_label']:15s} | ckpt={sr['checkpoint_mb']:6.1f}MB | params={sr['params_m']:6.2f}M | "
            f"model_vram={sr['model_vram_mb']:6.1f}MB | peak_alloc={sr['peak_alloc_mb']:6.1f}MB | "
            f"peak_res={sr['peak_reserved_mb']:6.1f}MB | infer={sr['total_infer_sec']:6.2f}s "
            f"({sr['mean_infer_sec']:.3f}s/img) | speedup={sr['speedup_vs_sam2']}x"
        )
    print(f"\nSaved summary:   {summary_csv}")
    print(f"Saved per-image: {per_img_csv}")
    print(f"Saved grids:     {out_root / 'comparison_grids'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Zero-Shot Model Benchmark")
    parser.add_argument("--run-model", type=str, default=None, choices=[k for k, _ in MODEL_SPECS])
    parser.add_argument("--all", action="store_true", help="Run all models sequentially in isolated processes")
    parser.add_argument("--images", type=Path, default=AI_DIR.parent / "images")
    parser.add_argument("--output", type=Path, default=AI_DIR / "benchmark_results")
    parser.add_argument("--weights-dir", type=Path, default=AI_DIR / "weights")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--only", type=str, nargs="*", default=None)
    args = parser.parse_args()

    if args.run_model:
        run_single_model_benchmark(
            model_key=args.run_model,
            images_dir=args.images,
            out_root=args.output,
            weights_dir=args.weights_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            only=args.only,
        )
        return

    # Default or --all: launch isolated subprocess per model for clean CUDA context & VRAM stats
    args.output.mkdir(parents=True, exist_ok=True)
    for model_key, model_label in MODEL_SPECS:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-model",
            model_key,
            "--images",
            str(args.images),
            "--output",
            str(args.output),
            "--weights-dir",
            str(args.weights_dir),
            "--tile-size",
            str(args.tile_size),
            "--overlap",
            str(args.overlap),
        ]
        if args.only:
            cmd.extend(["--only", *args.only])
        print(f"\n>>> Launching isolated subprocess for {model_label} ({model_key})...")
        subprocess.run(cmd, check=True)

    aggregate_reports(args.images, args.output, args.only)


if __name__ == "__main__":
    main()
