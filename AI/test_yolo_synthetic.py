from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

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


def deduplicate_mask_detections(dets: list[dict]) -> list[dict]:
    """
    Suppress cross-tile duplicates using pixel mask IoU and containment only.
    Avoids axis-aligned bbox IoU suppression so crossed screws are preserved.
    """
    if not dets:
        return []

    dets = [d for d in dets if d["area"] >= 60]
    if len(dets) >= 5:
        median_area = float(np.median([d["area"] for d in dets]))
        dets = [
            d
            for d in dets
            if not (d["is_border_clipped"] and d["area"] < 0.25 * median_area)
            and d["area"] >= 0.15 * median_area
        ]

    # Prefer intact interior detections over tile-border-clipped fragments, then higher confidence
    dets.sort(key=lambda d: (not d["is_border_clipped"], d["conf"]), reverse=True)
    kept: list[dict] = []

    for cand in dets:
        cx0, cy0, cx1, cy1 = cand["bbox"]
        c_area = cand["area"]
        suppressed = False

        for k in kept:
            kx0, ky0, kx1, ky1 = k["bbox"]
            ix0 = max(cx0, kx0)
            iy0 = max(cy0, ky0)
            ix1 = min(cx1, kx1)
            iy1 = min(cy1, ky1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue

            c_slice = cand["mask"][iy0 - cy0 : iy1 - cy0, ix0 - cx0 : ix1 - cx0] > 0
            k_slice = k["mask"][iy0 - ky0 : iy1 - ky0, ix0 - kx0 : ix1 - kx0] > 0
            mask_inter = int(np.count_nonzero(c_slice & k_slice))
            if mask_inter == 0:
                continue

            mask_iou = mask_inter / max(1, c_area + k["area"] - mask_inter)
            mask_contain = mask_inter / max(1, min(c_area, k["area"]))
            if mask_iou >= 0.35 or mask_contain >= 0.55:
                suppressed = True
                break

        if not suppressed:
            kept.append(cand)

    # Order deterministically top-to-bottom, left-to-right for clean instance numbering
    kept.sort(key=lambda d: (d["center"][1], d["center"][0]))
    return kept


def run_synthetic_yolo_on_image(
    model: YOLO,
    img_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    tile_size: int,
    overlap: int,
) -> tuple[list[dict], float, float]:
    """Run tiled YOLO11m-seg inference without classical contrast or hole heuristics."""
    h, w = img_bgr.shape[:2]
    t_pipe0 = time.perf_counter()
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
            conf=conf,
            iou=iou,
            imgsz=imgsz,
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

            # Flag only fragments truncated at an interior tile seam (not outer image borders)
            is_border_clipped = (
                (tx0 > 0 and (bx0 + ix0) <= 6)
                or (tx1 < w and (bx0 + ix1) >= tw - 6)
                or (ty0 > 0 and (by0 + iy0) <= 6)
                or (ty1 < h and (by0 + iy1) >= th - 6)
            )

            cx = int(round(tx0 + bx0 + xs.mean()))
            cy = int(round(ty0 + by0 + ys.mean()))

            raw_dets.append(
                {
                    "conf": float(confs[i]),
                    "bbox": (gx0, gy0, gx1, gy1),
                    "mask": tight_m,
                    "area": int(ys.size),
                    "center": (cx, cy),
                    "is_border_clipped": bool(is_border_clipped),
                }
            )

    dets = deduplicate_mask_detections(raw_dets)
    pipe_sec = time.perf_counter() - t_pipe0
    return dets, infer_sec, pipe_sec


def _make_panel_with_header(image: np.ndarray, header: str, panel_w: int, panel_h: int) -> np.ndarray:
    """Resize panel to (panel_w, panel_h) and prepend a crisp 48px black header strip."""
    resized = cv2.resize(image, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    bar_h = 48
    bar = np.zeros((bar_h, panel_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.80
    thick = 2

    # Shrink font slightly if header exceeds panel width
    (tw, th), _ = cv2.getTextSize(header, font, font_scale, thick)
    if tw > panel_w - 24:
        font_scale *= (panel_w - 24) / float(tw)
        (tw, th), _ = cv2.getTextSize(header, font, font_scale, thick)

    y_pos = (bar_h + th) // 2
    cv2.putText(bar, header, (12, y_pos), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
    return np.vstack([bar, resized])


def build_three_panel_grid(
    orig_bgr: np.ndarray,
    sam2_overlay_path: Path,
    yolo_overlay_bgr: np.ndarray,
    sam_entry: dict | None,
    yolo_count: int,
    yolo_infer_sec: float,
    panel_w: int,
    out_path: Path,
) -> None:
    """Create Original | SAM 2.1-b (Zero-Shot) | YOLO11m-seg (Synthetic) 3-panel grid."""
    oh, ow = orig_bgr.shape[:2]
    panel_h = int(round(panel_w * oh / ow))

    if sam2_overlay_path.exists():
        sam_img = cv2.imread(str(sam2_overlay_path))
        if sam_img is None:
            warnings.warn(f"Failed to read SAM 2 overlay {sam2_overlay_path}; using original.")
            sam_img = orig_bgr.copy()
    else:
        warnings.warn(f"SAM 2 overlay missing at {sam2_overlay_path}; using original.")
        sam_img = orig_bgr.copy()

    if sam_entry is not None:
        sam_header = f"SAM 2.1-b (Zero-Shot) | Count: {sam_entry['count']} | {sam_entry['infer_sec']:.2f}s"
    else:
        sam_header = "SAM 2.1-b (Zero-Shot) | N/A"

    p_orig = _make_panel_with_header(orig_bgr, f"Original ({ow}x{oh})", panel_w, panel_h)
    p_sam = _make_panel_with_header(sam_img, sam_header, panel_w, panel_h)
    p_yolo = _make_panel_with_header(
        yolo_overlay_bgr,
        f"YOLO11m-seg (Synthetic) | Count: {yolo_count} | {yolo_infer_sec:.2f}s",
        panel_w,
        panel_h,
    )

    sep = np.full((panel_h + 48, 4, 3), (60, 60, 60), dtype=np.uint8)
    grid = np.hstack([p_orig, sep, p_sam, sep, p_yolo])
    save_png(out_path, grid)


def load_sam2_metrics(sam2_dir: Path) -> tuple[dict | None, dict[str, dict]]:
    metrics_path = sam2_dir / "metrics.json"
    if not metrics_path.exists():
        warnings.warn(f"SAM 2 metrics not found at {metrics_path}")
        return None, {}
    data = json.loads(metrics_path.read_text())
    by_image = {entry["image"]: entry for entry in data.get("per_image", [])}
    return data, by_image


def load_nonai_counts(csv_path: Path) -> dict[str, int]:
    if not csv_path.exists():
        return {}
    mapping: dict[str, int] = {}
    with csv_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            val = row.get("final_adaptive", "").strip()
            if val:
                mapping[row["image"].strip()] = int(val)
    return mapping


def evaluate_model(
    weights_path: Path,
    images_dir: Path,
    sam2_dir: Path,
    output_dir: Path,
    tile_size: int,
    overlap: int,
    imgsz: int,
    conf: float,
    iou: float,
    panel_w: int,
    only: list[str] | None,
) -> None:
    weights_path = weights_path.resolve()
    images_dir = images_dir.resolve()
    sam2_dir = sam2_dir.resolve()
    output_dir = output_dir.resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"Trained weights not found at: {weights_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found at: {images_dir}")

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if only:
        only_set = set(only)
        image_paths = [p for p in image_paths if p.stem in only_set or p.name in only_set]
    if not image_paths:
        raise FileNotFoundError(f"No matching images found in {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sam2_summary, sam2_by_image = load_sam2_metrics(sam2_dir)
    nonai_map = load_nonai_counts(AI_DIR.parent / "Non_AI" / "adaptive_results" / "summary_counts.csv")

    model = YOLO(str(weights_path))
    ckpt_mb = weights_path.stat().st_size / (1024 ** 2)
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6

    # Warmup on first tile so VRAM and latency measurements exclude initialization overhead
    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        raise RuntimeError(f"Failed to read image: {image_paths[0]}")
    warmup_tile = first_img[: min(tile_size, first_img.shape[0]), : min(tile_size, first_img.shape[1])]
    model.predict(
        source=warmup_tile,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        retina_masks=True,
        max_det=300,
        device=0,
        verbose=False,
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    model_vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    torch.cuda.reset_peak_memory_stats(0)

    print(
        f"\n=== [YOLO11m-seg (Synthetic)] weights={weights_path.name} ({ckpt_mb:.1f}MB, {params_m:.2f}M) | "
        f"model_vram={model_vram_mb:.1f}MB | conf={conf} iou={iou} tile={tile_size} imgsz={imgsz} ==="
    )

    per_image: list[dict] = []
    comparison_rows: list[dict] = []

    for img_path in image_paths:
        stem = img_path.stem
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        h, w = img_bgr.shape[:2]

        dets, infer_sec, pipe_sec = run_synthetic_yolo_on_image(
            model=model,
            img_bgr=img_bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            tile_size=tile_size,
            overlap=overlap,
        )

        subtitle = f"YOLO11m-seg (Synthetic) | Infer: {infer_sec:.2f}s"
        yolo_overlay = render_yolo_overlay(img_bgr, dets, subtitle)
        save_png(output_dir / "overlays" / f"{stem}.png", yolo_overlay)
        save_label_mask((h, w), dets, output_dir / "label_masks" / f"{stem}_labels.png")

        sam_entry = sam2_by_image.get(img_path.name)
        build_three_panel_grid(
            orig_bgr=img_bgr,
            sam2_overlay_path=sam2_dir / "overlays" / f"{stem}.png",
            yolo_overlay_bgr=yolo_overlay,
            sam_entry=sam_entry,
            yolo_count=len(dets),
            yolo_infer_sec=infer_sec,
            panel_w=panel_w,
            out_path=output_dir / "comparison_grids" / f"{stem}_compare.png",
        )

        per_image.append(
            {
                "image": img_path.name,
                "split": "val" if stem in VAL_STEMS else "train",
                "count": len(dets),
                "infer_sec": round(infer_sec, 4),
                "pipe_sec": round(pipe_sec, 4),
            }
        )

        sam_cnt = sam_entry["count"] if sam_entry else None
        sam_inf = sam_entry["infer_sec"] if sam_entry else None
        diff_vs_sam2 = (len(dets) - sam_cnt) if sam_cnt is not None else ""
        speedup = round(sam_inf / infer_sec, 2) if (sam_inf is not None and infer_sec > 0) else ""

        comparison_rows.append(
            {
                "image": img_path.name,
                "non_ai_adaptive": nonai_map.get(img_path.name, ""),
                "sam2_count": sam_cnt if sam_cnt is not None else "",
                "yolo_synthetic_count": len(dets),
                "diff_vs_sam2": diff_vs_sam2,
                "sam2_infer_sec": sam_inf if sam_inf is not None else "",
                "yolo_infer_sec": round(infer_sec, 4),
                "speedup_vs_sam2": speedup,
                "yolo_pipe_sec": round(pipe_sec, 4),
            }
        )

        sam_str = f"{sam_cnt:3d}" if sam_cnt is not None else "N/A"
        diff_str = f"{diff_vs_sam2:+4d}" if isinstance(diff_vs_sam2, int) else " N/A"
        spd_str = f"{speedup:5.1f}x" if isinstance(speedup, float) else "  N/A"
        print(
            f"  [{img_path.name:20s}] yolo={len(dets):3d} | sam2={sam_str} | diff={diff_str} | "
            f"infer={infer_sec:5.3f}s | pipe={pipe_sec:5.3f}s | speedup={spd_str}"
        )

    torch.cuda.synchronize()
    peak_alloc_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    peak_res_mb = torch.cuda.max_memory_reserved(0) / (1024 ** 2)

    infer_times = [r["infer_sec"] for r in per_image]
    pipe_times = [r["pipe_sec"] for r in per_image]
    total_infer_sec = round(float(np.sum(infer_times)), 2)
    mean_infer_sec = round(float(np.mean(infer_times)), 3)
    total_pipe_sec = round(float(np.sum(pipe_times)), 2)
    mean_pipe_sec = round(float(np.mean(pipe_times)), 3)

    metrics = {
        "model_key": "yolo11m_synthetic_seg",
        "model_label": "YOLO11m-seg (Synthetic)",
        "weights": str(weights_path),
        "num_images": len(per_image),
        "checkpoint_mb": round(ckpt_mb, 2),
        "params_m": round(params_m, 2),
        "model_vram_mb": round(model_vram_mb, 1),
        "peak_alloc_mb": round(peak_alloc_mb, 1),
        "peak_reserved_mb": round(peak_res_mb, 1),
        "total_infer_sec": total_infer_sec,
        "mean_infer_sec": mean_infer_sec,
        "total_pipe_sec": total_pipe_sec,
        "mean_pipe_sec": mean_pipe_sec,
        "per_image": per_image,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    per_img_csv = output_dir / "comparison_per_image.csv"
    with per_img_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary_rows: list[dict] = []
    sam_total_infer = None
    if sam2_summary is not None:
        sam_total_infer = sam2_summary.get("total_infer_sec")
        summary_rows.append(
            {
                "model_key": sam2_summary["model_key"],
                "model_label": f"{sam2_summary['model_label']} (Zero-Shot)",
                "num_images": sam2_summary["num_images"],
                "checkpoint_mb": sam2_summary["checkpoint_mb"],
                "params_m": sam2_summary["params_m"],
                "model_vram_mb": sam2_summary["model_vram_mb"],
                "peak_alloc_mb": sam2_summary["peak_alloc_mb"],
                "peak_reserved_mb": sam2_summary["peak_reserved_mb"],
                "total_infer_sec": sam2_summary["total_infer_sec"],
                "mean_infer_sec": sam2_summary["mean_infer_sec"],
                "total_pipe_sec": sam2_summary["total_pipe_sec"],
                "mean_pipe_sec": sam2_summary["mean_pipe_sec"],
                "speedup_vs_sam2": 1.0,
            }
        )

    yolo_speedup = (
        round(sam_total_infer / total_infer_sec, 2)
        if (sam_total_infer and total_infer_sec > 0)
        else ""
    )
    summary_rows.append(
        {
            "model_key": metrics["model_key"],
            "model_label": metrics["model_label"],
            "num_images": metrics["num_images"],
            "checkpoint_mb": metrics["checkpoint_mb"],
            "params_m": metrics["params_m"],
            "model_vram_mb": metrics["model_vram_mb"],
            "peak_alloc_mb": metrics["peak_alloc_mb"],
            "peak_reserved_mb": metrics["peak_reserved_mb"],
            "total_infer_sec": metrics["total_infer_sec"],
            "mean_infer_sec": metrics["mean_infer_sec"],
            "total_pipe_sec": metrics["total_pipe_sec"],
            "mean_pipe_sec": metrics["mean_pipe_sec"],
            "speedup_vs_sam2": yolo_speedup,
        }
    )

    summary_csv = output_dir / "comparison_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n================ YOLO11m-seg (Synthetic) vs SAM 2.1-b ================")
    for sr in summary_rows:
        print(
            f"{sr['model_label']:25s} | ckpt={sr['checkpoint_mb']:6.1f}MB | params={sr['params_m']:6.2f}M | "
            f"model_vram={sr['model_vram_mb']:6.1f}MB | peak_alloc={sr['peak_alloc_mb']:6.1f}MB | "
            f"peak_res={sr['peak_reserved_mb']:6.1f}MB | infer={sr['total_infer_sec']:6.2f}s "
            f"({sr['mean_infer_sec']:.3f}s/img) | pipe={sr['total_pipe_sec']:6.2f}s | speedup={sr['speedup_vs_sam2']}x"
        )
    print(f"\nSaved metrics:          {output_dir / 'metrics.json'}")
    print(f"Saved comparison CSVs:  {summary_csv}")
    print(f"                        {per_img_csv}")
    print(f"Saved overlays:         {output_dir / 'overlays'}")
    print(f"Saved comparison grids: {output_dir / 'comparison_grids'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic-trained YOLO11m-seg best.pt on images/ and compare against SAM 2.1"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=AI_DIR / "runs" / "yolo11m_synthetic_seg" / "weights" / "best.pt",
    )
    parser.add_argument("--images", type=Path, default=AI_DIR.parent / "images")
    parser.add_argument("--sam2-dir", type=Path, default=AI_DIR / "benchmark_results" / "sam2_1_b")
    parser.add_argument("--output", type=Path, default=AI_DIR / "results" / "yolo11m_synthetic_eval")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--panel-w", type=int, default=900)
    parser.add_argument("--only", type=str, nargs="*", default=None)
    args = parser.parse_args()

    evaluate_model(
        weights_path=args.weights,
        images_dir=args.images,
        sam2_dir=args.sam2_dir,
        output_dir=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        panel_w=args.panel_w,
        only=args.only,
    )


if __name__ == "__main__":
    main()
