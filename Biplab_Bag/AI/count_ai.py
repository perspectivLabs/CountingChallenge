"""AI object counting + instance-mask overlay with a fine-tuned YOLO11-seg model.

    python count_ai.py -i <images or folders> -o outputs [--weights weights/best.pt] [--conf 0.35]

Each image is resized to the working resolution (long side 2000 px) and
segmented in one pass.  The model was trained on 1024 px tiles resized to
--train-imgsz (default 512), so the whole image is run at
imgsz = 2000 * train_imgsz / 1024 = 1000 -> 992 (multiple of 32) - the same object scale as in
training.  Count = number of predicted instances.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WORK_SIDE = 2000


def list_images(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() in IMG_EXT:
            out.append(p)
    return out


def resize_long_side(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def predict(model: YOLO, img: np.ndarray, conf: float, iou: float, imgsz: int):
    """Return list of (polygon Nx2 float, score) in image coordinates."""
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, max_det=3000, verbose=False, retina_masks=False)[0]
    if r.masks is None:
        return []
    return [(p, float(s)) for p, s in zip(r.masks.xy, r.boxes.conf.cpu().numpy()) if len(p) >= 3]


def rasterise(dets, shape) -> np.ndarray:
    """Instance label map; higher-confidence instances are drawn on top."""
    lab = np.zeros(shape, np.int32)
    order = np.argsort([s for _, s in dets])  # low -> high confidence
    for j in order:
        cv2.fillPoly(lab, [dets[j][0].astype(np.int32)], int(j + 1))
    return lab


def overlay(img: np.ndarray, labels: np.ndarray, dets, title: str) -> np.ndarray:
    n = len(dets)
    rng = np.random.default_rng(7)
    colors = rng.integers(40, 255, size=(n + 1, 3), dtype=np.uint8)
    colors[0] = 0
    fg = labels > 0
    out = img.copy()
    out[fg] = (0.45 * img[fg] + 0.55 * colors[labels][fg]).astype(np.uint8)
    for poly, _ in dets:
        cv2.polylines(out, [poly.astype(np.int32)], True, (0, 0, 0), 1, cv2.LINE_AA)
    fs = max(0.3, img.shape[1] / 4500)
    for j, (poly, _) in enumerate(dets, 1):
        cx, cy = poly.mean(0)
        cv2.putText(out, str(j), (int(cx) - 6, int(cy) + 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), 1, cv2.LINE_AA)
    sc = img.shape[1] / 1400
    cv2.rectangle(out, (0, 0), (img.shape[1], int(60 * sc)), (255, 255, 255), -1)
    cv2.putText(out, f"{title}  |  AI count = {n}", (int(15 * sc), int(42 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1 * sc, (0, 0, 200), max(1, int(2 * sc)), cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", nargs="+", required=True)
    ap.add_argument("--output", "-o", default="outputs")
    ap.add_argument("--weights", default=str(HERE / "weights" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--train-imgsz", type=int, default=512,
                    help="imgsz used in training. Training tiles are 1024 px, so the model saw objects at "
                         "train_imgsz/1024 of the working scale; inference reproduces that scale.")
    ap.add_argument("--imgsz", type=int, default=0, help="full-image inference size (0 = auto from --train-imgsz)")
    ap.add_argument("--save-masks", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.weights)
    scale = args.train_imgsz / 1024                     # object scale the model was trained at
    if not args.imgsz:
        args.imgsz = int(round(WORK_SIDE * scale / 32)) * 32
    print(f"training scale {scale:.2f} -> full image @ imgsz {args.imgsz}")
    out_dir = Path(args.output)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    images = list_images(args.input)
    if not images:
        raise SystemExit("no images found")
    rows = []
    for p in images:
        t0 = time.time()
        img = cv2.imread(str(p))
        if img is None:
            print(f"cannot read {p}, skipped")
            continue
        img = resize_long_side(img, WORK_SIDE)
        dets = predict(model, img, args.conf, args.iou, args.imgsz)
        labels = rasterise(dets, img.shape[:2])
        dt = time.time() - t0
        cv2.imwrite(str(out_dir / "overlays" / f"{p.stem}_overlay.jpg"), overlay(img, labels, dets, p.name),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        if args.save_masks:
            cv2.imwrite(str(out_dir / "masks" / f"{p.stem}_labels.png"), labels.astype(np.uint16))
        rows.append({"image": p.name, "count": len(dets), "seconds": round(dt, 2)})
        print(f"{p.name:28s} count={len(dets):4d}  {dt:.2f}s")
    with open(out_dir / "counts.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nSaved {len(rows)} results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
