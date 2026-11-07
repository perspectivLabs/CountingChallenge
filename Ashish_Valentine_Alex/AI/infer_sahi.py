"""SAHI sliced inference with the trained YOLO26-seg model: count items and overlay masks on full-resolution photos.

    python infer_sahi.py --model count_item.pt --input <folder with the original photos> --output <out folder>

Each photo is cut into 1024x1024 slices (20% overlap), the model runs on every slice, the masks are mapped back to the
original photo's coordinates and duplicates from overlapping slices are merged (mask NMS, IOS metric; defaults conf 0.35 / merge 0.2 tuned on the val photos). Writes, per photo,
an overlay jpg with the count, and one counts.csv (+ polygons.json with the mask outlines in full-image pixels).
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from shapely.geometry import Polygon
from shapely.strtree import STRtree

TILE, OVERLAP = 1024, 0.20


def load_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")     # phone photos carry EXIF rotation


def predict(det, pil, conf, match_thr):
    """Run the model on every slice, keep detections with score >= conf, then merge duplicates from overlapping slices.
    Returns a list of (score, segmentation) with segmentation = list of flat polygons in full-image pixels."""
    det.confidence_threshold = conf
    r = get_sliced_prediction(pil, det, slice_height=TILE, slice_width=TILE,
                              overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
                              perform_standard_pred=False, postprocess_type="NMS",
                              postprocess_match_metric="IOS", postprocess_match_threshold=1.01,   # no merge here; done below
                              postprocess_class_agnostic=True, verbose=0)
    return merge([(q.score.value, q.mask.segmentation) for q in r.object_prediction_list if q.mask is not None], match_thr)


def merge(items, match_thr):
    """Greedy NMS on the mask polygons, highest score first. Two masks are the same item when their overlap area is more
    than match_thr of the smaller mask (IOS), which also removes a small partial mask lying on top of a full one."""
    polys = []
    for score, segs in items:
        ps = [Polygon([(g[i], g[i + 1]) for i in range(0, len(g) - 1, 2)]) for g in segs if len(g) >= 6]
        if not ps: continue
        p = max(ps, key=lambda q: q.area)
        p = p if p.is_valid else p.buffer(0)
        if not p.is_empty and p.area > 20: polys.append((score, segs, p))
    polys.sort(key=lambda x: -x[0])
    tree = STRtree([p for _, _, p in polys]); dropped, kept = set(), []
    for i, (score, segs, p) in enumerate(polys):
        if i in dropped: continue
        kept.append((score, segs))
        for j in tree.query(p):
            if j > i and j not in dropped and p.intersection(polys[j][2]).area / min(p.area, polys[j][2].area) > match_thr:
                dropped.add(j)
    return kept


def overlay(pil, preds, out_path, max_side):
    scale = min(1.0, max_side / max(pil.size)) if max_side else 1.0
    base = pil.resize((int(pil.width * scale), int(pil.height * scale))) if scale < 1 else pil.copy()
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    rng = np.random.RandomState(1)
    for _, segs in preds:
        col = tuple(int(c) for c in rng.randint(60, 255, 3))
        for poly in segs:
            pts = [(poly[i] * scale, poly[i + 1] * scale) for i in range(0, len(poly) - 1, 2)]
            if len(pts) >= 3: d.polygon(pts, fill=col + (110,), outline=col + (255,))
    out = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    ImageDraw.Draw(out).text((20, 20), f"count = {len(preds)}", fill=(255, 0, 0))
    out.save(out_path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--conf", type=float, default=0.35); ap.add_argument("--match", type=float, default=0.2, help="merge threshold (IOS)")
    ap.add_argument("--device", default=None, help="cuda:0 or cpu (default: cuda if available)")
    ap.add_argument("--max-side", type=int, default=0, help="downscale overlays to this long side (0 = full resolution)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N photos (quick test)")
    a = ap.parse_args()
    if a.device is None:
        import torch; a.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = Path(a.output); (out / "overlays").mkdir(parents=True, exist_ok=True)
    det = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=a.model,
                                             confidence_threshold=a.conf, device=a.device, image_size=TILE)
    photos = [p for p in sorted(Path(a.input).rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if a.limit: photos = photos[:a.limit]
    rows, polys = [], {}
    for p in photos:
        pil = load_rgb(p); preds = predict(det, pil, a.conf, a.match)
        overlay(pil, preds, out / "overlays" / f"{p.stem}.jpg", a.max_side)
        rows.append((p.stem, len(preds))); polys[p.stem] = [q[1] for q in preds]
        print(f"{p.stem:22s} count={len(preds)}", flush=True)
    with open(out / "counts.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "count"]); w.writerows(rows)
    json.dump(polys, open(out / "polygons.json", "w"))


if __name__ == "__main__":
    main()
