from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, YOLOE

AI_DIR = Path(__file__).resolve().parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from prepare_dataset import VAL_STEMS, generate_tiles  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]
YOLOE_PROMPTS = ["screw", "hex nut"]
CLASS_NAMES = {0: "screw", 1: "nut"}


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, PNG_PARAMS)


def draw_corner_badge(
    img: np.ndarray,
    text: str,
    subtitle: str | None = None,
) -> np.ndarray:
    """Draw a resolution-scaled count badge in the top-left corner."""
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


def render_yolo_overlay(
    img_bgr: np.ndarray,
    detections: list[dict],
    model_label: str,
) -> np.ndarray:
    """Render semi-transparent per-instance masks, crisp borders, IDs, and corner count badge."""
    h, w = img_bgr.shape[:2]
    scale = max(h, w) / 1500.0
    line_thick = max(1, int(round(1.8 * scale)))
    font_scale = max(0.38, 0.52 * scale)
    if len(detections) > 120:
        font_scale *= 0.62
    elif len(detections) > 50:
        font_scale *= 0.80
    font_thick = max(1, int(round(1.5 * scale)))

    color_mask = np.zeros_like(img_bgr)
    rng = np.random.default_rng(seed=42)

    for det in detections:
        col = tuple(int(c) for c in rng.integers(60, 255, size=3))
        x0, y0, x1, y1 = det["bbox"]
        roi_cm = color_mask[y0:y1, x0:x1]
        roi_cm[det["mask"] > 0] = col

    overlay = cv2.addWeighted(img_bgr, 0.58, color_mask, 0.42, 0)

    border_col = (0, 0, 255)  # Uniform crisp red contour for all instances
    for idx, det in enumerate(detections, start=1):
        x0, y0, x1, y1 = det["bbox"]
        cnts, _ = cv2.findContours(det["mask"], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        offset = np.array([[[x0, y0]]], dtype=np.int32)
        shifted = [c + offset for c in cnts]
        cv2.drawContours(overlay, shifted, -1, border_col, line_thick, cv2.LINE_AA)

        cx, cy = det["center"]
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

    title = f"Count: {len(detections)}"
    return draw_corner_badge(overlay, title, model_label)


def save_label_mask(img_shape: tuple[int, int], detections: list[dict], out_path: Path) -> None:
    """Save full-resolution 16-bit instance label mask PNG (0=bg, 1..N=instance ID)."""
    h, w = img_shape
    label_map = np.zeros((h, w), dtype=np.uint16)
    for idx, det in enumerate(detections, start=1):
        x0, y0, x1, y1 = det["bbox"]
        roi = label_map[y0:y1, x0:x1]
        roi[det["mask"] > 0] = idx
    save_png(out_path, label_map)


def run_tiled_inference(
    model: YOLO | YOLOE,
    img_bgr: np.ndarray,
    conf: float,
    tile_size: int = 1024,
    overlap: int = 256,
) -> list[dict]:
    """Run sliding-window YOLO-seg inference across tiles and deduplicate overlapping masks."""
    h, w = img_bgr.shape[:2]
    tiles = generate_tiles(h, w, tile_size=tile_size, overlap=overlap)
    raw_dets: list[dict] = []

    for tx0, ty0, tx1, ty1 in tiles:
        tile_img = img_bgr[ty0:ty1, tx0:tx1]
        th, tw = tile_img.shape[:2]

        results = model.predict(
            source=tile_img,
            conf=conf,
            iou=0.55,
            imgsz=640,
            retina_masks=True,
            max_det=300,
            device=0,
            verbose=False,
        )
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

            # Tighten bounding box to actual foreground pixels
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
            cx = int(round(tx0 + bx0 + xs.mean()))
            cy = int(round(ty0 + by0 + ys.mean()))

            raw_dets.append(
                {
                    "cls_id": cls_id,
                    "cls_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                    "conf": float(confs[i]),
                    "bbox": (gx0, gy0, gx1, gy1),
                    "mask": tight_m,
                    "area": int(ys.size),
                    "center": (cx, cy),
                    "is_border_clipped": bool(is_border_clipped),
                }
            )

    return deduplicate_detections(raw_dets)


def deduplicate_detections(dets: list[dict]) -> list[dict]:
    """Suppress duplicate detections across overlapping tile borders using box & mask overlap."""
    if not dets:
        return []

    dets = [d for d in dets if d["area"] >= 60]
    if len(dets) >= 5:
        median_area = float(np.median([d["area"] for d in dets]))
        cls_medians: dict[int, float] = {}
        for cid in {d["cls_id"] for d in dets}:
            c_areas = [d["area"] for d in dets if d["cls_id"] == cid]
            if len(c_areas) >= 5:
                cls_medians[cid] = float(np.median(c_areas))
        dets = [
            d for d in dets
            if not (d["is_border_clipped"] and d["area"] < 0.25 * median_area)
            and d["area"] >= 0.15 * cls_medians.get(d["cls_id"], 0.0)
        ]

    # Prefer intact interior detections over tile-border-clipped halves, then higher confidence
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

            box_inter = (ix1 - ix0) * (iy1 - iy0)
            c_box_area = (cx1 - cx0) * (cy1 - cy0)
            k_box_area = (kx1 - kx0) * (ky1 - ky0)
            box_iou = box_inter / max(1, c_box_area + k_box_area - box_inter)
            if box_iou >= 0.45:
                suppressed = True
                break

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

    # Order deterministically top-to-bottom, left-to-right for clean numbering
    kept.sort(key=lambda d: (d["center"][1], d["center"][0]))
    return kept


def load_yoloe_model(weights_dir: Path) -> YOLOE:
    weights_dir.mkdir(parents=True, exist_ok=True)
    prev_cwd = Path.cwd()
    try:
        # Keep any Ultralytics auto-downloaded files (yoloe-11l-seg.pt, mobileclip_blt.ts) inside AI/weights/
        os.chdir(weights_dir)
        model = YOLOE(str(weights_dir / "yoloe-11l-seg.pt"))
        text_pe = model.get_text_pe(YOLOE_PROMPTS)
        model.set_classes(YOLOE_PROMPTS, text_pe)
    finally:
        os.chdir(prev_cwd)
    return model


def load_finetuned_model(weights_path: Path) -> YOLO:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned weights not found at {weights_path}. Run train_yolo_seg.py first."
        )
    return YOLO(str(weights_path))


def load_nonai_counts(summary_csv: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not summary_csv.exists():
        return counts
    with summary_csv.open("r", newline="") as f:
        for row in csv.DictReader(f):
            counts[row["image"].strip()] = int(row["final_adaptive"])
    return counts


def run_pipeline(
    mode: str,
    images_dir: Path,
    out_root: Path,
    weights_dir: Path,
    conf_finetuned: float,
    conf_yoloe: float,
    only: list[str] | None = None,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if only:
        only_set = set(only)
        image_paths = [p for p in image_paths if p.stem in only_set or p.name in only_set]

    yoloe_model = load_yoloe_model(weights_dir) if mode in ("yoloe", "both") else None
    finetuned_model = (
        load_finetuned_model(weights_dir / "yolo11s_screw_nut_seg.pt")
        if mode in ("finetuned", "both")
        else None
    )

    nonai_csv = AI_DIR.parent / "Non_AI" / "adaptive_results" / "summary_counts.csv"
    nonai_counts = load_nonai_counts(nonai_csv)

    print(f"Running YOLO segmentation (mode={mode}) on {len(image_paths)} images -> {out_root}\n")
    rows: list[dict] = []
    t_total = time.perf_counter()

    for img_path in image_paths:
        stem = img_path.stem
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        h, w = img_bgr.shape[:2]

        row: dict[str, object] = {
            "image": img_path.name,
            "split": "val" if stem in VAL_STEMS else "train",
            "non_ai_adaptive": nonai_counts.get(img_path.name, ""),
        }

        if yoloe_model is not None:
            t0 = time.perf_counter()
            dets_ye = run_tiled_inference(yoloe_model, img_bgr, conf=conf_yoloe)
            ov_ye = render_yolo_overlay(img_bgr, dets_ye, f"Zero-Shot YOLOE-11L-seg (conf={conf_yoloe})")
            save_png(out_root / "yoloe_zeroshot" / "overlays" / f"{stem}.png", ov_ye)
            row["yoloe_count"] = len(dets_ye)
            row["yoloe_sec"] = round(time.perf_counter() - t0, 2)

        if finetuned_model is not None:
            t0 = time.perf_counter()
            dets_ft = run_tiled_inference(finetuned_model, img_bgr, conf=conf_finetuned)
            ov_ft = render_yolo_overlay(img_bgr, dets_ft, f"Fine-Tuned YOLO11s-seg (conf={conf_finetuned})")
            save_png(out_root / "yolo11_finetuned" / "overlays" / f"{stem}.png", ov_ft)
            save_label_mask(
                (h, w),
                dets_ft,
                out_root / "yolo11_finetuned" / "label_masks" / f"{stem}_labels.png",
            )
            row["yolo11_count"] = len(dets_ft)
            row["yolo11_sec"] = round(time.perf_counter() - t0, 2)

        rows.append(row)

        parts = [f"[{img_path.name:20s}] ({row['split']:5s}) NonAI={str(row['non_ai_adaptive']):>3s}"]
        if "yoloe_count" in row:
            parts.append(f"YOLOE={row['yoloe_count']:3d}")
        if "yolo11_count" in row:
            parts.append(f"YOLO11-FT={row['yolo11_count']:3d}")
        print(" | ".join(parts))

    if rows:
        csv_path = out_root / "comparison_counts.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDone in {time.perf_counter() - t_total:.2f}s. Saved comparison CSV to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO-Seg Screw & Nut Counting Pipeline")
    parser.add_argument(
        "--mode",
        choices=["yoloe", "finetuned", "both"],
        default="both",
        help="Run zero-shot YOLOE-seg, fine-tuned YOLO11-seg, or both",
    )
    parser.add_argument("--images", type=Path, default=AI_DIR.parent / "images")
    parser.add_argument("--output", type=Path, default=AI_DIR / "results")
    parser.add_argument("--weights-dir", type=Path, default=AI_DIR / "weights")
    parser.add_argument("--conf-finetuned", type=float, default=0.30)
    parser.add_argument("--conf-yoloe", type=float, default=0.18)
    parser.add_argument("--only", type=str, nargs="*", default=None)
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        images_dir=args.images,
        out_root=args.output,
        weights_dir=args.weights_dir,
        conf_finetuned=args.conf_finetuned,
        conf_yoloe=args.conf_yoloe,
        only=args.only,
    )


if __name__ == "__main__":
    main()
