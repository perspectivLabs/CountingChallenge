# """Non-AI screw / bolt / nut counting, one phase at a time.

#     python phases.py --input data/ --output outputs --upto 8
#     python phases.py --input data/img01.jpg --output outputs --upto 3

# --input   an image file, or a folder of images (any format OpenCV can read, any extension)
# --output  folder where every phase gets its own sub-folder
# --upto    stop after this phase (1-8); default 8 = run everything

# Each phase saves  <output>/<NN_phase_name>/<original file name>  with the SAME name, extension
# and size as the input, and hands its result to the next phase.
# """
# from __future__ import annotations

# import argparse
# from pathlib import Path

# import cv2
# import numpy as np
# import pandas as pd
# from scipy import ndimage as ndi
# from skimage.morphology import h_maxima
# from skimage.segmentation import find_boundaries, watershed

# # ----------------------------------------------------------------------------- settings
# # Pixel values are for the ORIGINAL image size. If your images are very large (> ~2500 px),
# # scale the kernel sizes up or downscale the image first.
# P = dict(
#     # phase 1
#     bilateral_d=7, sigma_color=40, sigma_space=7,
#     # phase 2
#     border_frac=0.03,      # border strip (x short side) used to measure the background colour
#     l_weight=0.5,          # lightness weight in LAB distance (<1 = ignore shadows more)
#     blur_sigma=1.5,        # smooth the distance map before thresholding
#     # phase 3
#     thresh_scale=1.0,      # >1 = stricter foreground, <1 = more permissive
#     # phase 4
#     median_k=5, open_k=3, close_k=5,
#     max_hole_frac=0.0005,  # holes smaller than this x image area are filled (glare), bigger stay
#     min_blob_px=40,
#     # phase 6
#     clean_solidity=0.90,   # blobs this convex define the "typical single object"
#     split_solidity=0.85,   # blobs below this are candidates for splitting
#     split_area_ratio=1.6,  # blobs above this x typical area are candidates for splitting
#     smooth_sigma=1.5,      # smoothing of the distance map before finding seeds
#     h_frac=0.20,           # seed height threshold, as a fraction of the typical half-width
#     min_piece_ratio=0.25,  # watershed pieces smaller than this x typical area get merged back
#     merge_ratio=1.30,      # neighbouring pieces whose total area <= this x typical are re-merged
#     # phase 7
#     min_area_ratio=0.20, max_area_ratio=8.0,
#     nut_max_aspect=1.6,    # length/width below this -> "nut", otherwise "screw/bolt"
#     unsure_solidity=0.70,  # blobs less convex than this are probably merged objects -> "unsure"
#     # phase 8
#     alpha=0.45,
# )


# # ----------------------------------------------------------------------------- helpers
# LABEL_MAP_EXTS = {".png", ".tif", ".tiff"}  # formats that store 16-bit object ids losslessly


# def read_image(path: Path):
#     """Read any format OpenCV understands; None if the file is not a readable image.
#     (fromfile + imdecode also works for non-ASCII paths on Windows.)"""
#     try:
#         data = np.fromfile(str(path), dtype=np.uint8)
#         return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
#     except OSError:
#         return None


# def write_image(path: Path, img: np.ndarray) -> None:
#     """Write img to `path`; the file format is chosen from the path's own extension."""
#     try:
#         ok, buf = cv2.imencode(path.suffix, img)
#     except cv2.error:
#         ok = False
#     if not ok:
#         raise ValueError(f"OpenCV cannot write '{path.suffix}' files")
#     buf.tofile(str(path))


# def save(out_dir: Path, phase: str, name: str, img: np.ndarray) -> Path:
#     """Save into <out_dir>/<phase>/<name>; `name` is the original file name (extension included)."""
#     folder = out_dir / phase
#     folder.mkdir(parents=True, exist_ok=True)
#     path = folder / name
#     write_image(path, img)
#     return path


# def palette(n: int) -> np.ndarray:
#     """n+1 BGR colours; index 0 (background) is black."""
#     hues = (np.arange(n + 1) * 0.61803398875) % 1.0
#     hsv = np.stack([hues * 179, np.full(n + 1, 210), np.full(n + 1, 255)], 1).astype(np.uint8)[None]
#     pal = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
#     pal[0] = 0
#     return pal


# def blend(img, labels, colors, alpha=0.45, ids=False):
#     """Colour every labelled region (colors[label]), outline it, optionally write its id."""
#     out = img.copy()
#     m = labels > 0
#     col = colors[labels]
#     out[m] = ((1 - alpha) * img[m] + alpha * col[m]).astype(np.uint8)
#     edge = find_boundaries(labels, mode="inner")
#     out[edge] = (0.35 * out[edge] + 0.65 * col[edge]).astype(np.uint8)
#     if ids:
#         for lab, sl in enumerate(ndi.find_objects(labels), start=1):
#             if sl is None:
#                 continue
#             cy, cx = ndi.center_of_mass(labels[sl] == lab)
#             cv2.putText(out, str(lab), (int(sl[1].start + cx) - 8, int(sl[0].start + cy) + 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
#     return out


# def banner(img, lines):
#     out = img.copy()
#     s = max(img.shape[:2]) / 1000.0
#     fs, th, pad = max(s, 0.6), max(2, int(2 * s)), int(10 * max(s, 0.6))
#     sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0] for t in lines]
#     lh = max(h for _, h in sizes) + pad
#     cv2.rectangle(out, (0, 0), (max(w for w, _ in sizes) + 2 * pad, lh * len(lines) + pad), (0, 0, 0), -1)
#     for i, t in enumerate(lines):
#         cv2.putText(out, t, (pad, pad + lh * (i + 1) - pad // 2), cv2.FONT_HERSHEY_SIMPLEX,
#                     fs, (255, 255, 255), th, cv2.LINE_AA)
#     return out


# def fill_small_holes(mask, max_area):
#     """Fill enclosed holes up to max_area pixels (glare). Bigger holes, e.g. a nut's centre, stay."""
#     m = mask > 0
#     holes = (ndi.binary_fill_holes(m) & ~m).astype(np.uint8)
#     n, lab, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
#     small = np.zeros(n, bool)
#     small[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_area
#     return (m | small[lab]).astype(np.uint8) * 255


# def describe(labels):
#     """One row per labelled blob: area, solidity, length/width/aspect, bbox, touches_border.
#     Solidity and aspect use the OUTER contour, so a nut's hole does not hurt them."""
#     H, W = labels.shape
#     rows = []
#     for lab, sl in enumerate(ndi.find_objects(labels), start=1):
#         if sl is None:
#             continue
#         sub = (labels[sl] == lab).astype(np.uint8)
#         cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         c = max(cnts, key=cv2.contourArea)
#         hull_area = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
#         (_, _), (rw, rh), _ = cv2.minAreaRect(c)
#         length, width = max(rw, rh), max(min(rw, rh), 1e-6)
#         y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
#         rows.append(dict(id=lab, area=int(sub.sum()), solidity=cv2.contourArea(c) / hull_area,
#                          length=length, width=width, aspect=length / width,
#                          x0=x0, y0=y0, x1=x1, y1=y1,
#                          touches_border=bool(x0 == 0 or y0 == 0 or x1 >= W or y1 >= H)))
#     return pd.DataFrame(rows)


# def estimate_unit(table):
#     """Typical single-object area = median area of blobs that are convex and not cut by the border."""
#     if table.empty:
#         return 0.0
#     clean = table[table.solidity >= P["clean_solidity"]]
#     pool = clean[~clean.touches_border]
#     if len(pool) < 3:
#         pool = clean if len(clean) else table
#     return float(pool.area.median())


# # ----------------------------------------------------------------------------- phases
# def phase1_preprocess(img):
#     """Denoise but keep edges (bilateral filter)."""
#     return cv2.bilateralFilter(img, P["bilateral_d"], P["sigma_color"], P["sigma_space"])


# def phase2_distance_map(img):
#     """Per-pixel LAB distance from the background colour (median of the border strip), 0-255."""
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
#     b = max(2, int(min(img.shape[:2]) * P["border_frac"]))
#     border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
#                              lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
#     bg = np.median(border, axis=0)
#     w = np.array([P["l_weight"], 1.0, 1.0], np.float32)
#     dist = np.sqrt((((lab - bg) * w) ** 2).sum(-1))
#     d8 = np.clip(dist / max(np.percentile(dist, 99.5), 1e-6) * 255, 0, 255).astype(np.uint8)
#     if P["blur_sigma"] > 0:
#         d8 = cv2.GaussianBlur(d8, (0, 0), P["blur_sigma"])
#     return d8, bg


# def phase3_threshold(d8):
#     """Otsu picks the threshold that best separates 'near background' from 'far from background'."""
#     t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     t_used = t * P["thresh_scale"]
#     return (d8 > t_used).astype(np.uint8) * 255, t, t_used


# def phase4_clean(mask):
#     """Median filter -> open (remove specks) -> close (bridge gaps) -> fill glare holes -> drop tiny blobs."""
#     if P["median_k"] > 1:
#         mask = cv2.medianBlur(mask, P["median_k"] | 1)
#     for op, size in ((cv2.MORPH_OPEN, P["open_k"]), (cv2.MORPH_CLOSE, P["close_k"])):
#         if size > 1:
#             mask = cv2.morphologyEx(mask, op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
#     mask = fill_small_holes(mask, int(P["max_hole_frac"] * mask.size))
#     n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
#     keep = np.zeros(n, bool)
#     keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= P["min_blob_px"]
#     return keep[lab].astype(np.uint8) * 255


# def phase5_components(mask):
#     """Give every separate blob its own id and measure it."""
#     _, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
#     labels = labels.astype(np.int32)
#     return labels, describe(labels)


# # ---- phase 6 helpers: splitting touching objects
# def _border_lengths(ws, i):
#     m = ws == i
#     ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
#     neigh = ws[ring]
#     neigh = neigh[neigh > 0]
#     return {} if neigh.size == 0 else {int(j): int(c) for j, c in enumerate(np.bincount(neigh)) if c > 0}


# def _merge_pieces(ws, min_piece, max_merge):
#     """Undo spurious splits: (1) tiny pieces join the neighbour they border most;
#     (2) neighbours whose COMBINED area is about one object are merged. Real pairs stay split."""
#     while True:
#         ids, counts = np.unique(ws[ws > 0], return_counts=True)
#         if len(ids) <= 1:
#             break
#         area = {int(i): int(c) for i, c in zip(ids, counts)}
#         best = None
#         for i in area:
#             for j, border in _border_lengths(ws, i).items():
#                 if j <= i:
#                     continue
#                 if min(area[i], area[j]) < min_piece:
#                     key = (0, -border)
#                 elif area[i] + area[j] <= max_merge:
#                     key = (1, area[i] + area[j])
#                 else:
#                     continue
#                 if best is None or key < best[0]:
#                     best = (key, i, j)
#         if best is None:
#             break
#         ws[ws == best[2]] = best[1]
#     return ws


# def _split_component(sub, h, min_piece, max_merge):
#     """sub = 0/1 crop of ONE blob. Returns (pieces, seeds). Holes are filled while splitting
#     (a nut's ring would otherwise give several seeds) and carved out again afterwards."""
#     filled = ndi.binary_fill_holes(sub).astype(np.uint8)
#     d = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
#     d = cv2.GaussianBlur(d, (0, 0), P["smooth_sigma"]) * filled
#     seeds = (h_maxima((d * 16).astype(np.int32), max(1, int(round(h * 16)))) > 0) & (filled > 0)
#     markers, k = ndi.label(seeds)
#     if k <= 1:
#         return sub.astype(np.int32), seeds
#     ws = watershed(-d, markers, mask=filled > 0)
#     ws = _merge_pieces(ws, min_piece, max_merge)
#     ws[sub == 0] = 0
#     return ws, seeds


# def phase6_split(labels, table):
#     """Split only suspicious blobs (not convex, or much bigger than typical)."""
#     H, W = labels.shape
#     out = np.zeros((H, W), np.int32)
#     seeds_full = np.zeros((H, W), np.uint8)
#     if table.empty:
#         return out, seeds_full, dict(unit_area=0.0, split=0)

#     unit_area = estimate_unit(table)
#     pool = table[(table.solidity >= P["clean_solidity"]) & ~table.touches_border]
#     if len(pool) < 3:
#         pool = table

#     def crop(r, pad=2):
#         y0, x0 = max(r.y0 - pad, 0), max(r.x0 - pad, 0)
#         y1, x1 = min(r.y1 + pad, H), min(r.x1 + pad, W)
#         return (labels[y0:y1, x0:x1] == r.id).astype(np.uint8), (y0, y1, x0, x1)

#     half_widths = []
#     for r in pool.itertuples():
#         sub, _ = crop(r)
#         half_widths.append(cv2.distanceTransform(ndi.binary_fill_holes(sub).astype(np.uint8),
#                                                  cv2.DIST_L2, 5).max())
#     h = max(1.0, P["h_frac"] * float(np.median(half_widths)))

#     next_id, n_split = 1, 0
#     for r in table.itertuples():
#         sub, (y0, y1, x0, x1) = crop(r)
#         pieces = sub.astype(np.int32)
#         if r.solidity < P["split_solidity"] or r.area > P["split_area_ratio"] * unit_area:
#             pieces, seeds = _split_component(sub, h, P["min_piece_ratio"] * unit_area,
#                                              P["merge_ratio"] * unit_area)
#             seeds_full[y0:y1, x0:x1] |= seeds.astype(np.uint8) * 255
#             n_split += len(np.unique(pieces[pieces > 0])) > 1
#         region = out[y0:y1, x0:x1]
#         for pid in np.unique(pieces[pieces > 0]):
#             region[pieces == pid] = next_id
#             next_id += 1
#     return out, seeds_full, dict(unit_area=unit_area, split=int(n_split))


# def phase7_filter_classify(labels):
#     """Drop blobs far smaller/larger than typical, then label each as nut or screw/bolt by shape."""
#     table = describe(labels)
#     unit = estimate_unit(table)
#     keep = table[(table.area >= max(P["min_area_ratio"] * unit, P["min_blob_px"]))
#                  & (table.area <= P["max_area_ratio"] * unit)].copy()
#     remap = np.zeros(int(labels.max()) + 1, np.int32)
#     remap[keep.id.values] = np.arange(1, len(keep) + 1)
#     keep["id"] = np.arange(1, len(keep) + 1)
#     keep["type"] = np.select([keep.solidity < P["unsure_solidity"], keep.aspect < P["nut_max_aspect"]],
#                              ["unsure", "nut"], default="screw/bolt")
#     return remap[labels], keep.reset_index(drop=True), unit


# # ----------------------------------------------------------------------------- main
# def run_phases(img, name, out, upto, show_table):
#     """Run phases 1..upto on one image. Returns the final counts if phase 8 ran, else None."""
#     pre = phase1_preprocess(img)
#     save(out, "01_preprocess", name, pre)
#     if upto < 2: return None

#     d8, bg = phase2_distance_map(pre)
#     save(out, "02_distance_map", name, cv2.applyColorMap(d8, cv2.COLORMAP_JET))
#     print(f"[2] background colour (LAB) = {np.round(bg, 1)}")
#     if upto < 3: return None

#     raw, t, t_used = phase3_threshold(d8)
#     save(out, "03_threshold", name, raw)
#     print(f"[3] Otsu threshold = {t:.0f} (used {t_used:.0f}); foreground = {(raw > 0).mean() * 100:.1f}%")
#     if upto < 4: return None

#     mask = phase4_clean(raw)
#     save(out, "04_clean_mask", name, mask)
#     print(f"[4] foreground after cleaning = {(mask > 0).mean() * 100:.1f}%")
#     if upto < 5: return None

#     labels, table = phase5_components(mask)
#     save(out, "05_components", name, blend(pre, labels, palette(int(labels.max())), P["alpha"], ids=True))
#     table.to_csv(out / "05_components" / f"{name}.csv", index=False)
#     print(f"[5] {len(table)} separate blobs")
#     if show_table and len(table):
#         print(table[["id", "area", "solidity", "aspect", "touches_border"]].round(2).to_string(index=False))
#     if upto < 6: return None

#     labels, seeds, info = phase6_split(labels, table)
#     vis = blend(pre, labels, palette(int(labels.max())), P["alpha"])
#     vis[cv2.dilate(seeds, np.ones((5, 5), np.uint8)) > 0] = (0, 0, 255)  # seeds in red
#     save(out, "06_split", name, vis)
#     print(f"[6] typical object area = {info['unit_area']:.0f}px; blobs split = {info['split']}; "
#           f"now {int(labels.max())} regions")
#     if upto < 7: return None

#     labels, table, unit = phase7_filter_classify(labels)
#     cls_color = {"nut": (0, 165, 255), "screw/bolt": (0, 200, 0), "unsure": (0, 0, 255)}  # BGR
#     colors = np.zeros((len(table) + 1, 3), np.uint8)
#     for r in table.itertuples():
#         colors[r.id] = cls_color[r.type]
#     n_nut = int((table.type == "nut").sum()) if len(table) else 0
#     n_unsure = int((table.type == "unsure").sum()) if len(table) else 0
#     n_sb = len(table) - n_nut - n_unsure
#     save(out, "07_filter_classify", name,
#          banner(blend(pre, labels, colors, P["alpha"]),
#                 [f"screw/bolt (green): {n_sb}", f"nut (orange): {n_nut}", f"unsure (red): {n_unsure}"]))
#     table.to_csv(out / "07_filter_classify" / f"{name}.csv", index=False)
#     print(f"[7] kept {len(table)} objects: {n_sb} screw/bolt, {n_nut} nut, {n_unsure} unsure")
#     if upto < 8: return None

#     n = len(table)
#     save(out, "08_overlay", name,
#          banner(blend(img, labels, palette(n), P["alpha"]),
#                 [f"Total: {n}", f"screw/bolt: {n_sb}  nut: {n_nut}  unsure: {n_unsure}"]))
#     # Object ids need a lossless 16-bit format; jpg/webp/bmp would corrupt them, so those get '.png' added.
#     lm_name = name if Path(name).suffix.lower() in LABEL_MAP_EXTS else name + ".png"
#     save(out, "08_label_map", lm_name, labels.astype(np.uint16))  # pixel value = object id
#     print(f"[8] FINAL COUNT = {n}")
#     return dict(count=n, screw_bolt=n_sb, nut=n_nut, unsure=n_unsure)


# def collect_inputs(path: Path) -> list[Path]:
#     if path.is_file():
#         return [path]
#     if path.is_dir():  # every regular file; non-images are skipped later when they fail to decode
#         return sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith("."))
#     raise SystemExit(f"Input not found: {path}")


# def main():
#     ap = argparse.ArgumentParser(description="Non-AI screw/bolt/nut counting, phase by phase.")
#     ap.add_argument("--input", required=True, type=Path,
#                     help="image file, or folder of images (any extension OpenCV can read)")
#     ap.add_argument("--output", required=True, type=Path,
#                     help="folder in which each phase gets its own sub-folder")
#     ap.add_argument("--upto", type=int, default=8, choices=range(1, 9), metavar="N",
#                     help="stop after phase N (1-8, default 8)")
#     a = ap.parse_args()

#     files = collect_inputs(a.input)
#     a.output.mkdir(parents=True, exist_ok=True)
#     rows, skipped, failed = [], [], []

#     for p in files:
#         img = read_image(p)
#         if img is None:
#             skipped.append(p.name)
#             continue
#         name = p.name if p.suffix else p.name + ".png"  # a file with no extension needs one to be saved
#         print(f"\n=== {p.name}  ({img.shape[1]}x{img.shape[0]})")
#         try:
#             res = run_phases(img, name, a.output, a.upto, show_table=(len(files) == 1))
#         except Exception as e:
#             if len(files) == 1:
#                 raise  # single image: show the full traceback
#             failed.append(p.name)
#             print(f"  FAILED: {type(e).__name__}: {e}")
#             continue
#         if res:
#             rows.append(dict(image=name, **res))

#     n_ok = len(files) - len(skipped) - len(failed)
#     if rows:
#         pd.DataFrame(rows).to_csv(a.output / "counts.csv", index=False)
#     print(f"\nDone: {n_ok} image(s) processed up to phase {a.upto} -> {a.output}/")
#     if rows:
#         print(f"Counts saved to {a.output / 'counts.csv'}")
#     if skipped:
#         print(f"Skipped (not readable as images): {', '.join(skipped)}")
#     if failed:
#         print(f"Failed: {', '.join(failed)}")
#     if n_ok == 0:
#         raise SystemExit("No readable images found.")


# if __name__ == "__main__":
#     main()

"""Non-AI screw / bolt / nut counting, one phase at a time.

    python phases.py --input data/ --output outputs --upto 8
    python phases.py --input data/img01.jpg --output outputs --upto 3

--input   an image file, or a folder of images (any format OpenCV can read, any extension)
--output  folder where every phase gets its own sub-folder
--upto    stop after this phase (1-8); default 8 = run everything

Each phase saves  <output>/<NN_phase_name>/<original file name>  with the SAME name, extension
and size as the input, and hands its result to the next phase.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import find_boundaries, watershed

# ----------------------------------------------------------------------------- settings
# Pixel values are for the ORIGINAL image size. If your images are very large (> ~2500 px),
# scale the kernel sizes up or downscale the image first.
P = dict(
    # phase 1
    bilateral_d=7, sigma_color=40, sigma_space=7,
    # phase 2
    border_frac=0.03,      # border strip (x short side) used to measure the background colour
    l_weight=0.5,          # lightness weight in LAB distance (<1 = ignore shadows more)
    blur_sigma=1.5,        # smooth the distance map before thresholding
    # phase 3
    thresh_scale=1.0,      # >1 = stricter foreground, <1 = more permissive
    # phase 4
    median_k=5, open_k=3, close_k=5,
    max_hole_frac=0.0005,  # holes smaller than this x image area are filled (glare), bigger stay
    min_blob_px=40,
    # phase 6  (ONE object class per image -> every object has about the same area)
    split_area_ratio=1.5,  # blobs bigger than this x typical object are clusters and get split
    max_blob_frac=0.30,    # blobs bigger than this fraction of the image are not objects (tray, leak)
    smooth_sigma=1.5,      # smoothing of the distance map before finding seeds
    h_frac=0.20,           # peak-seed height, as a fraction of the typical half-width
    erode_frac=0.5,        # erosion-seed radius, as a fraction of the typical half-width
    min_piece_ratio=0.25,  # pieces smaller than this x typical area are merged into a neighbour
    axis_tol=0.30,         # how closely a cluster must match "k objects in a row" (30% tolerance)
    # phase 7
    min_area_ratio=0.25,   # final pieces smaller than this x typical area are fragments -> dropped
    nut_max_aspect=1.6,    # typical length/width below this -> image is "nut", otherwise "screw/bolt"
    # phase 8
    alpha=0.45,
)


# ----------------------------------------------------------------------------- helpers
LABEL_MAP_EXTS = {".png", ".tif", ".tiff"}  # formats that store 16-bit object ids losslessly


def read_image(path: Path):
    """Read any format OpenCV understands; None if the file is not a readable image.
    (fromfile + imdecode also works for non-ASCII paths on Windows.)"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


def write_image(path: Path, img: np.ndarray) -> None:
    """Write img to `path`; the file format is chosen from the path's own extension."""
    try:
        ok, buf = cv2.imencode(path.suffix, img)
    except cv2.error:
        ok = False
    if not ok:
        raise ValueError(f"OpenCV cannot write '{path.suffix}' files")
    buf.tofile(str(path))


def save(out_dir: Path, phase: str, name: str, img: np.ndarray) -> Path:
    """Save into <out_dir>/<phase>/<name>; `name` is the original file name (extension included)."""
    folder = out_dir / phase
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    write_image(path, img)
    return path


def palette(n: int) -> np.ndarray:
    """n+1 BGR colours; index 0 (background) is black."""
    hues = (np.arange(n + 1) * 0.61803398875) % 1.0
    hsv = np.stack([hues * 179, np.full(n + 1, 210), np.full(n + 1, 255)], 1).astype(np.uint8)[None]
    pal = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
    pal[0] = 0
    return pal


def blend(img, labels, colors, alpha=0.45, ids=False):
    """Colour every labelled region (colors[label]), outline it, optionally write its id."""
    out = img.copy()
    m = labels > 0
    col = colors[labels]
    out[m] = ((1 - alpha) * img[m] + alpha * col[m]).astype(np.uint8)
    edge = find_boundaries(labels, mode="inner")
    out[edge] = (0.35 * out[edge] + 0.65 * col[edge]).astype(np.uint8)
    if ids:
        for lab, sl in enumerate(ndi.find_objects(labels), start=1):
            if sl is None:
                continue
            cy, cx = ndi.center_of_mass(labels[sl] == lab)
            cv2.putText(out, str(lab), (int(sl[1].start + cx) - 8, int(sl[0].start + cy) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def banner(img, lines):
    out = img.copy()
    s = max(img.shape[:2]) / 1000.0
    fs, th, pad = max(s, 0.6), max(2, int(2 * s)), int(10 * max(s, 0.6))
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0] for t in lines]
    lh = max(h for _, h in sizes) + pad
    cv2.rectangle(out, (0, 0), (max(w for w, _ in sizes) + 2 * pad, lh * len(lines) + pad), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (pad, pad + lh * (i + 1) - pad // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), th, cv2.LINE_AA)
    return out


def fill_small_holes(mask, max_area):
    """Fill enclosed holes up to max_area pixels (glare). Bigger holes, e.g. a nut's centre, stay."""
    m = mask > 0
    holes = (ndi.binary_fill_holes(m) & ~m).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_area
    return (m | small[lab]).astype(np.uint8) * 255


def describe(labels):
    """One row per labelled blob: area, solidity, length/width/aspect, bbox, touches_border.
    Solidity and aspect use the OUTER contour, so a nut's hole does not hurt them."""
    H, W = labels.shape
    rows = []
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        sub = (labels[sl] == lab).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        hull_area = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        length, width = max(rw, rh), max(min(rw, rh), 1e-6)
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        rows.append(dict(id=lab, area=int(sub.sum()), solidity=cv2.contourArea(c) / hull_area,
                         length=length, width=width, aspect=length / width,
                         x0=x0, y0=y0, x1=x1, y1=y1,
                         touches_border=bool(x0 == 0 or y0 == 0 or x1 >= W or y1 >= H)))
    return pd.DataFrame(rows)


def estimate_unit(table):
    """Typical single-object area. With one class per image all objects are about the same size,
    so the median blob area is a good start. It is then refined using only blobs between 0.5x and
    1.5x of the current estimate, which ignores clusters (too big) and fragments (too small)."""
    if table.empty:
        return 0.0
    t = table[~table.touches_border]
    if len(t) < 3:
        t = table
    a = t.area.to_numpy(dtype=float)
    unit = float(np.median(a))
    for _ in range(3):
        sel = a[(a >= 0.5 * unit) & (a <= 1.5 * unit)]
        if sel.size == 0:
            break
        unit = float(np.median(sel))
    return unit


def estimate_half_width(table, unit):
    """Typical half-width of one object = area / (2 * length), median over single-object blobs.
    (For a screw this is about the shank half-width, not the wider head.)"""
    singles = table[(table.area >= 0.5 * unit) & (table.area <= 1.5 * unit)]
    if singles.empty:
        singles = table
    return float(np.median(singles.area / (2.0 * singles.length.clip(lower=1.0))))


# ----------------------------------------------------------------------------- phases
def phase1_preprocess(img):
    """Denoise but keep edges (bilateral filter)."""
    return cv2.bilateralFilter(img, P["bilateral_d"], P["sigma_color"], P["sigma_space"])


def phase2_distance_map(img):
    """Per-pixel LAB distance from the background colour (median of the border strip), 0-255."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(2, int(min(img.shape[:2]) * P["border_frac"]))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    w = np.array([P["l_weight"], 1.0, 1.0], np.float32)
    dist = np.sqrt((((lab - bg) * w) ** 2).sum(-1))
    d8 = np.clip(dist / max(np.percentile(dist, 99.5), 1e-6) * 255, 0, 255).astype(np.uint8)
    if P["blur_sigma"] > 0:
        d8 = cv2.GaussianBlur(d8, (0, 0), P["blur_sigma"])
    return d8, bg


def phase3_threshold(d8):
    """Otsu picks the threshold that best separates 'near background' from 'far from background'."""
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_used = t * P["thresh_scale"]
    return (d8 > t_used).astype(np.uint8) * 255, t, t_used


def phase4_clean(mask):
    """Median filter -> open (remove specks) -> close (bridge gaps) -> fill glare holes -> drop tiny blobs."""
    if P["median_k"] > 1:
        mask = cv2.medianBlur(mask, P["median_k"] | 1)
    for op, size in ((cv2.MORPH_OPEN, P["open_k"]), (cv2.MORPH_CLOSE, P["close_k"])):
        if size > 1:
            mask = cv2.morphologyEx(mask, op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    mask = fill_small_holes(mask, int(P["max_hole_frac"] * mask.size))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= P["min_blob_px"]
    return keep[lab].astype(np.uint8) * 255


def phase5_components(mask):
    """Give every separate blob its own id and measure it."""
    _, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    labels = labels.astype(np.int32)
    return labels, describe(labels)


# ---- phase 6 helpers: splitting clusters into k objects
def _border_lengths(ws, i):
    """For piece i: {neighbour id: number of touching pixels}."""
    m = ws == i
    ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
    neigh = ws[ring]
    neigh = neigh[neigh > 0]
    return {} if neigh.size == 0 else {int(j): int(c) for j, c in enumerate(np.bincount(neigh)) if c > 0}


def _merge_to_k(ws, k, min_piece):
    """(1) Tiny pieces join the neighbour they share the longest border with.
    (2) While there are more than k pieces, merge the touching pair with the smallest combined area.
        A screw cut into head + shank is the smallest pair, so it is rejoined first."""
    while True:
        ids, counts = np.unique(ws[ws > 0], return_counts=True)
        if len(ids) <= 1:
            break
        area = dict(zip(ids.tolist(), counts.tolist()))
        best = None
        for i in area:
            for j, border in _border_lengths(ws, i).items():
                if j <= i:
                    continue
                if min(area[i], area[j]) < min_piece:
                    key = (0, -border)
                elif len(ids) > k:
                    key = (1, area[i] + area[j])
                else:
                    continue
                if best is None or key < best[0]:
                    best = (key, i, j)
        if best is None:
            break
        ws[ws == best[2]] = best[1]
    return ws


def _seeds_erosion(filled, radius, min_px):
    """Shrink the blob: thin bridges between touching objects disappear, each object keeps a core."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.erode(filled, k), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    lab[~keep[lab]] = 0
    return lab


def _seeds_peaks(filled, d, h):
    """Peaks of the distance map that stand out by at least h (h-maxima)."""
    seeds = (h_maxima((d * 16).astype(np.int32), max(1, int(round(h * 16)))) > 0) & (filled > 0)
    return ndi.label(seeds)[0]


def _kmeans_split(sub, k):
    """Last resort: cut the blob's pixels into k spatial groups. The count is right,
    the mask borders are only approximate."""
    pts = np.column_stack(np.nonzero(sub)).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, lab, _ = cv2.kmeans(pts, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    out = np.zeros(sub.shape, np.int32)
    out[sub > 0] = lab.ravel() + 1
    return out


def _axis_split(sub, k, obj_len, obj_wid):
    """Objects lying in a row: k side by side, or k end to end. Measure the cluster along its main
    axis (PCA). If it is ~k objects long and ~1 wide (end to end), or ~1 long and ~k wide (side by
    side), cut it into k equal-area slices across that direction. Returns None if it is neither."""
    ys, xs = np.nonzero(sub)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov(pts.T))
    major, minor = pts @ vecs[:, 1], pts @ vecs[:, 0]
    ext_major, ext_minor = np.ptp(major), np.ptp(minor)
    close = lambda value, target: abs(value / max(target, 1e-6) - 1) <= P["axis_tol"]
    if close(ext_major, k * obj_len) and close(ext_minor, obj_wid):
        coord = major        # end to end
    elif close(ext_major, obj_len) and close(ext_minor, k * obj_wid):
        coord = minor        # side by side
    else:
        return None
    cuts = np.quantile(coord, np.arange(1, k) / k)
    out = np.zeros(sub.shape, np.int32)
    out[ys, xs] = np.digitize(coord, cuts) + 1
    return out


def _split_cluster(sub, k, unit, half_w, obj_len, obj_wid):
    """Split one cluster blob into k objects.
    First checks for objects lying in a row (side by side / end to end). Otherwise tries erosion
    seeds and peak seeds, merges each result down to k pieces, and keeps the one
    that reaches k with the most even piece sizes. If neither reaches k, falls back to k-means."""
    row = _axis_split(sub, k, obj_len, obj_wid)
    if row is not None:
        return row, np.zeros(sub.shape, bool), "row"

    filled = ndi.binary_fill_holes(sub).astype(np.uint8)   # nut holes would create extra seeds
    d = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    d = cv2.GaussianBlur(d, (0, 0), P["smooth_sigma"]) * filled
    min_piece = P["min_piece_ratio"] * unit
    radius = max(1, int(round(P["erode_frac"] * half_w)))

    best = None
    for method, markers in (("erosion", _seeds_erosion(filled, radius, max(4, int(0.02 * unit)))),
                            ("peaks", _seeds_peaks(filled, d, P["h_frac"] * half_w))):
        if markers.max() < 2:
            continue
        ws = watershed(-d, markers, mask=filled > 0)
        ws = _merge_to_k(ws, k, min_piece)
        ws[sub == 0] = 0
        _, counts = np.unique(ws[ws > 0], return_counts=True)
        score = (abs(len(counts) - k), float(np.std(counts) / max(np.mean(counts), 1.0)))
        if best is None or score < best[0]:
            best = (score, method, ws, markers > 0)

    if best is None or len(np.unique(best[2][best[2] > 0])) < k:
        return _kmeans_split(sub, k), np.zeros(sub.shape, bool), "kmeans"
    return best[2], best[3], best[1]


def phase6_split(labels, table):
    """Single class per image -> a blob's area tells how many objects it holds: k = round(area / unit).
    Blobs below split_area_ratio x unit are ONE object and are never split, however irregular their
    shape is (this is what keeps a screw's head and shank together). Bigger blobs are split into k."""
    H, W = labels.shape
    out = np.zeros((H, W), np.int32)
    seeds_full = np.zeros((H, W), np.uint8)
    info = dict(unit_area=0.0, half_width=0.0, length=0.0, width=0.0, clusters=[])
    if table.empty:
        return out, seeds_full, info

    unit = estimate_unit(table)
    half_w = estimate_half_width(table, unit)
    singles = table[(table.area >= 0.5 * unit) & (table.area <= 1.5 * unit)]
    singles = singles if len(singles) else table
    obj_len, obj_wid = float(singles.length.median()), float(singles.width.median())
    info.update(unit_area=unit, half_width=half_w, length=obj_len, width=obj_wid)

    next_id = 1
    for r in table.itertuples():
        if r.area > P["max_blob_frac"] * H * W:
            continue  # far too big to be an object: tray edge, background leak
        y0, x0 = max(r.y0 - 2, 0), max(r.x0 - 2, 0)
        y1, x1 = min(r.y1 + 2, H), min(r.x1 + 2, W)
        sub = (labels[y0:y1, x0:x1] == r.id).astype(np.uint8)
        pieces = sub.astype(np.int32)

        if r.area > P["split_area_ratio"] * unit:
            k = max(2, int(round(r.area / unit)))
            pieces, seeds, method = _split_cluster(sub, k, unit, half_w, obj_len, obj_wid)
            seeds_full[y0:y1, x0:x1] |= seeds.astype(np.uint8) * 255
            got = len(np.unique(pieces[pieces > 0]))
            info["clusters"].append(dict(blob=r.id, area=r.area, expected=k, got=got, method=method))

        region = out[y0:y1, x0:x1]
        for pid in np.unique(pieces[pieces > 0]):
            region[pieces == pid] = next_id
            next_id += 1
    return out, seeds_full, info


def phase7_filter_classify(labels, unit):
    """Drop fragments, then name the image's single class from the typical object shape."""
    table = describe(labels)
    if table.empty:
        return labels, table, "none"
    keep = table[table.area >= max(P["min_area_ratio"] * unit, P["min_blob_px"])].copy()
    remap = np.zeros(int(labels.max()) + 1, np.int32)
    remap[keep.id.values] = np.arange(1, len(keep) + 1)
    keep["id"] = np.arange(1, len(keep) + 1)
    singles = keep[(keep.area >= 0.6 * unit) & (keep.area <= 1.4 * unit)]
    ref = singles if len(singles) else keep
    cls = "nut" if float(ref.aspect.median()) < P["nut_max_aspect"] else "screw/bolt"
    keep["type"] = cls
    return remap[labels], keep.reset_index(drop=True), cls


# ----------------------------------------------------------------------------- main
def run_phases(img, name, out, upto, show_table):
    """Run phases 1..upto on one image. Returns the final counts if phase 8 ran, else None."""
    pre = phase1_preprocess(img)
    save(out, "01_preprocess", name, pre)
    if upto < 2: return None

    d8, bg = phase2_distance_map(pre)
    save(out, "02_distance_map", name, cv2.applyColorMap(d8, cv2.COLORMAP_JET))
    print(f"[2] background colour (LAB) = {np.round(bg, 1)}")
    if upto < 3: return None

    raw, t, t_used = phase3_threshold(d8)
    save(out, "03_threshold", name, raw)
    print(f"[3] Otsu threshold = {t:.0f} (used {t_used:.0f}); foreground = {(raw > 0).mean() * 100:.1f}%")
    if upto < 4: return None

    mask = phase4_clean(raw)
    save(out, "04_clean_mask", name, mask)
    print(f"[4] foreground after cleaning = {(mask > 0).mean() * 100:.1f}%")
    if upto < 5: return None

    labels, table = phase5_components(mask)
    save(out, "05_components", name, blend(pre, labels, palette(int(labels.max())), P["alpha"], ids=True))
    table.to_csv(out / "05_components" / f"{name}.csv", index=False)
    print(f"[5] {len(table)} separate blobs")
    if show_table and len(table):
        print(table[["id", "area", "solidity", "aspect", "touches_border"]].round(2).to_string(index=False))
    if upto < 6: return None

    labels, seeds, info = phase6_split(labels, table)
    vis = blend(pre, labels, palette(int(labels.max())), P["alpha"])
    vis[cv2.dilate(seeds, np.ones((5, 5), np.uint8)) > 0] = (0, 0, 255)  # seeds in red
    save(out, "06_split", name, vis)
    print(f"[6] typical object: area = {info['unit_area']:.0f}px, length x width = "
          f"{info['length']:.0f} x {info['width']:.0f}px, half-width = {info['half_width']:.1f}px; "
          f"clusters split = {len(info['clusters'])}; now {int(labels.max())} regions")
    for c in info["clusters"]:
        flag = "" if c["got"] == c["expected"] else "   <- check"
        print(f"    blob {c['blob']}: area {c['area']} -> expected {c['expected']}, got {c['got']} "
              f"({c['method']}){flag}")
    if upto < 7: return None

    labels, table, cls = phase7_filter_classify(labels, info["unit_area"])
    n = len(table)
    save(out, "07_filter_classify", name,
         banner(blend(pre, labels, palette(n), P["alpha"], ids=True), [f"class: {cls}", f"objects: {n}"]))
    table.to_csv(out / "07_filter_classify" / f"{name}.csv", index=False)
    print(f"[7] class = {cls}; kept {n} objects")
    if upto < 8: return None

    save(out, "08_overlay", name, banner(blend(img, labels, palette(n), P["alpha"]), [f"{cls}: {n}"]))
    # Object ids need a lossless 16-bit format; jpg/webp/bmp would corrupt them, so those get '.png' added.
    lm_name = name if Path(name).suffix.lower() in LABEL_MAP_EXTS else name + ".png"
    save(out, "08_label_map", lm_name, labels.astype(np.uint16))  # pixel value = object id
    print(f"[8] FINAL COUNT = {n}")
    return dict(count=n, cls=cls, clusters_split=len(info["clusters"]),
                kmeans_used=sum(c["method"] == "kmeans" for c in info["clusters"]))


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():  # every regular file; non-images are skipped later when they fail to decode
        return sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith("."))
    raise SystemExit(f"Input not found: {path}")


def main():
    ap = argparse.ArgumentParser(description="Non-AI screw/bolt/nut counting, phase by phase.")
    ap.add_argument("--input", required=True, type=Path,
                    help="image file, or folder of images (any extension OpenCV can read)")
    ap.add_argument("--output", required=True, type=Path,
                    help="folder in which each phase gets its own sub-folder")
    ap.add_argument("--upto", type=int, default=8, choices=range(1, 9), metavar="N",
                    help="stop after phase N (1-8, default 8)")
    a = ap.parse_args()

    files = collect_inputs(a.input)
    a.output.mkdir(parents=True, exist_ok=True)
    rows, skipped, failed = [], [], []

    for p in files:
        img = read_image(p)
        if img is None:
            skipped.append(p.name)
            continue
        name = p.name if p.suffix else p.name + ".png"  # a file with no extension needs one to be saved
        print(f"\n=== {p.name}  ({img.shape[1]}x{img.shape[0]})")
        try:
            res = run_phases(img, name, a.output, a.upto, show_table=(len(files) == 1))
        except Exception as e:
            if len(files) == 1:
                raise  # single image: show the full traceback
            failed.append(p.name)
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        if res:
            rows.append(dict(image=name, **res))

    n_ok = len(files) - len(skipped) - len(failed)
    if rows:
        pd.DataFrame(rows).to_csv(a.output / "counts.csv", index=False)
    print(f"\nDone: {n_ok} image(s) processed up to phase {a.upto} -> {a.output}/")
    if rows:
        print(f"Counts saved to {a.output / 'counts.csv'}")
    if skipped:
        print(f"Skipped (not readable as images): {', '.join(skipped)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    if n_ok == 0:
        raise SystemExit("No readable images found.")


if __name__ == "__main__":
    main()