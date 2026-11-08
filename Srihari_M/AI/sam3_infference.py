#  python3 sam3_infference.py --images ./Sample_data/Sample_data/ --output ./SAM3_inference --model ./sam3.pt --text screw nut --prompt-conf screw=0.35 nut=0.1 --imgsz 1190 --save-binary-masks --overlap 0.2 --nms-iou 0.4
"""
Batch SAM3 text-prompt segmentation with optional SAHI-style slicing
and per-class / total object counting.

Outputs (inside --output):
    BBOX_images/             image with boxes + count summary drawn on it
    MASK_images/             image with coloured mask overlays
    MASK_binary/             (only with --save-binary-masks) uint16 instance-id PNGs
    Annotation_JSON_files/   boxes, confidences, labels, polygons, counts
    counts_summary.csv       per-image counts + an ALL_IMAGES totals row

Example:
    python sam3_batch_count.py --images ./imgs --output ./out --model sam3.pt \
        --text "person" "car" --use-sahi --slice 1008 --overlap 0.2
"""

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics.models.sam import SAM3SemanticPredictor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class SAM3BatchProcessor:
    """Batch SAM3 processor with optional SAHI-style slicing and object counting."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        half: bool = True,
        imgsz: int = 1008,
        use_sahi: bool = False,
        slice_size: int = 1008,
        overlap_ratio: float = 0.2,
        nms_iou: float = 0.35,
        cross_class_nms: bool = False,
        save_binary_masks: bool = False,
        prompt_conf: Optional[Dict[str, float]] = None,
        debug: bool = False,
    ):
        self.conf = conf
        self.prompt_conf = dict(prompt_conf or {})
        self.debug = debug
        # run the model at the lowest threshold any prompt needs; filter per prompt later
        floor_conf = min([conf, *self.prompt_conf.values()])
        overrides = dict(
            model=model_path,
            task="segment",
            mode="predict",
            conf=floor_conf,
            half=half,
            imgsz=imgsz,
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

        self.use_sahi = use_sahi
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.nms_iou = nms_iou
        self.cross_class_nms = cross_class_nms
        self.save_binary_masks = save_binary_masks

        if use_sahi:
            print(f"[INFO] SAHI enabled | slice={slice_size} | overlap={overlap_ratio}")

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    @staticmethod
    def sanitize_filename(text: str) -> str:
        return "".join(
            c for c in text.replace(" ", "_").lower() if c.isalnum() or c in "_-"
        )

    def make_tag(self, prompts: List[str]) -> str:
        tag = "_".join(self.sanitize_filename(p) for p in prompts)
        return tag[:80]  # keep filenames at a safe length

    def setup_output_dirs(self, out_dir: str) -> Dict[str, Path]:
        base = Path(out_dir)
        dirs = {
            "bbox": base / "BBOX_images",
            "mask": base / "MASK_images",
            "json": base / "Annotation_JSON_files",
        }
        if self.save_binary_masks:
            dirs["binary"] = base / "MASK_binary"
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    @staticmethod
    def resolve_label(cid: int, class_names: Any, prompts: List[str]) -> str:
        """Map a class id to a readable label (result names -> prompt -> fallback)."""
        if isinstance(class_names, dict) and class_names.get(cid):
            return str(class_names[cid])
        if isinstance(class_names, (list, tuple)) and 0 <= cid < len(class_names):
            return str(class_names[cid])
        if 0 <= cid < len(prompts):
            return prompts[cid]
        return f"class_{cid}"

    # ------------------------------------------------------------------ #
    # SAHI slicing
    # ------------------------------------------------------------------ #

    def slice_image(self, image: np.ndarray) -> List[Tuple[np.ndarray, int, int]]:
        h, w = image.shape[:2]
        step = max(1, int(self.slice_size * (1 - self.overlap_ratio)))

        slices, seen = [], set()
        for y in range(0, h, step):
            for x in range(0, w, step):
                x2 = min(x + self.slice_size, w)
                y2 = min(y + self.slice_size, h)
                x1 = max(0, x2 - self.slice_size)
                y1 = max(0, y2 - self.slice_size)
                if (x1, y1) in seen:  # edge slices can repeat; skip duplicates
                    continue
                seen.add((x1, y1))
                slices.append((image[y1:y2, x1:x2], x1, y1))
        return slices

    # ------------------------------------------------------------------ #
    # NMS
    # ------------------------------------------------------------------ #

    @staticmethod
    def nms(boxes, scores, iou_thr):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)

        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]

        return keep

    # ------------------------------------------------------------------ #
    # Prediction (SAHI or single pass)
    # ------------------------------------------------------------------ #

    def run_prediction(
        self, image: np.ndarray, image_path: str, prompts: List[str]
    ) -> List[Tuple[Any, int, int, int]]:
        """Return (result, x_offset, y_offset, prompt_index) tuples.

        Each prompt is queried separately, so a detection's class is exactly
        the index of the prompt that produced it (no reliance on r.names).
        """
        outputs = []

        def query(path: str, xoff: int, yoff: int) -> None:
            self.predictor.set_image(path)
            for pi, prompt in enumerate(prompts):
                outputs.append((self.predictor(text=[prompt]), xoff, yoff, pi))

        if not self.use_sahi:
            query(image_path, 0, 0)
            return outputs

        tmp = Path(tempfile.mkdtemp())
        try:
            for i, (simg, x, y) in enumerate(self.slice_image(image)):
                p = tmp / f"{i}.jpg"
                cv2.imwrite(str(p), simg)
                query(str(p), x, y)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return outputs

    # ------------------------------------------------------------------ #
    # Merge slice results (tiling-aware de-duplication)
    # ------------------------------------------------------------------ #
    #
    # Problems this solves when an image is tiled:
    #   1. Same object seen whole in two overlapping tiles      -> duplicate
    #   2. Object cut by a tile edge, whole in another tile     -> stray fragment
    #   3. Object cut by tile edges in every tile (long object) -> several fragments
    #
    # A detection is "cut" if its box touches a slice edge that is NOT the
    # image border. Whole detections are ranked first; then
    #   (1) a detection is dropped if it overlaps a kept one with
    #       IoU > nms_iou (mask IoU when masks exist, else box IoU);
    #   (2) a cut fragment is dropped if it lies inside a kept whole object;
    #   (3) remaining cut fragments of the same class that overlap each other
    #       across a seam are merged into one detection (boxes + masks united).
    # Mask overlap is used whenever masks are available, so two different
    # objects with overlapping boxes (e.g. crossing screws) are NOT merged.

    EDGE_EPS = 3           # px: box this close to an interior slice edge counts as cut
    FRAG_MASK_IOS = 0.5    # fragment inside whole object: mask overlap / fragment area
    FRAG_BOX_IOS = 0.8     # same test when masks are unavailable (boxes only)
    MERGE_MASK_IOS = 0.1   # two fragments across a seam: mask overlap / smaller area
    MERGE_BOX_IOS = 0.3    # same test when masks are unavailable

    @staticmethod
    def _box_inter(a, b) -> float:
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return max(0.0, w) * max(0.0, h)

    @staticmethod
    def _box_area(b) -> float:
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    @staticmethod
    def _paste(members, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """OR member masks into a canvas covering [x1:x2, y1:y2] (full-image coords)."""
        canvas = np.zeros((max(0, y2 - y1), max(0, x2 - x1)), dtype=np.uint8)
        for m, xo, yo in members:
            mh, mw = m.shape
            ax1, ay1 = max(xo, x1), max(yo, y1)
            ax2, ay2 = min(xo + mw, x2), min(yo + mh, y2)
            if ax2 <= ax1 or ay2 <= ay1:
                continue
            canvas[ay1 - y1 : ay2 - y1, ax1 - x1 : ax2 - x1] |= m[
                ay1 - yo : ay2 - yo, ax1 - xo : ax2 - xo
            ]
        return canvas

    def _set_area(self, d: Dict[str, Any]) -> None:
        if not d["members"]:
            d["area"] = self._box_area(d["box"])
            return
        x1 = min(xo for _, xo, _ in d["members"])
        y1 = min(yo for _, _, yo in d["members"])
        x2 = max(xo + m.shape[1] for m, xo, _ in d["members"])
        y2 = max(yo + m.shape[0] for m, _, yo in d["members"])
        d["area"] = float(self._paste(d["members"], x1, y1, x2, y2).sum())

    def _overlap(self, a: Dict[str, Any], b: Dict[str, Any]):
        """Return (iou, ios, used_masks). ios = intersection / smaller area."""
        inter_box = self._box_inter(a["box"], b["box"])
        if inter_box <= 0:
            return 0.0, 0.0, False

        if a["members"] and b["members"]:
            x1 = int(np.floor(max(a["box"][0], b["box"][0])))
            y1 = int(np.floor(max(a["box"][1], b["box"][1])))
            x2 = int(np.ceil(min(a["box"][2], b["box"][2])))
            y2 = int(np.ceil(min(a["box"][3], b["box"][3])))
            ca = self._paste(a["members"], x1, y1, x2, y2)
            cb = self._paste(b["members"], x1, y1, x2, y2)
            inter = float((ca & cb).sum())
            area_a, area_b, used_masks = a["area"], b["area"], True
        else:
            inter = inter_box
            area_a, area_b = self._box_area(a["box"]), self._box_area(b["box"])
            used_masks = False

        union = area_a + area_b - inter
        smaller = min(area_a, area_b)
        return inter / (union + 1e-6), inter / (smaller + 1e-6), used_masks

    def merge_slice_results(
        self, slice_outputs, image_shape, prompts: List[str]
    ) -> Dict[str, Any]:
        H, W = image_shape[:2]
        dets: List[Dict[str, Any]] = []
        class_names = {i: p for i, p in enumerate(prompts)}
        stats = {p: {"raw": 0, "max_conf": 0.0, "passed": 0} for p in prompts}

        # ---- 1. collect detections from every slice (full-image coordinates) ----
        for result, xoff, yoff, pi in slice_outputs:
            if not result or len(result) == 0:
                continue

            r = result[0]

            if r.boxes is None or len(r.boxes) == 0:
                continue

            sh, sw = getattr(r, "orig_shape", None) or (self.slice_size, self.slice_size)

            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            thr = self.prompt_conf.get(prompts[pi], self.conf)
            st = stats[prompts[pi]]
            st["raw"] += len(xyxy)
            st["max_conf"] = max(st["max_conf"], float(conf.max()))

            slice_masks = None
            if r.masks is not None:
                m_all = r.masks.data.cpu().numpy()
                if len(m_all) == len(xyxy):
                    slice_masks = m_all

            for i in range(len(xyxy)):
                if float(conf[i]) < thr:
                    continue
                st["passed"] += 1
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                eps = self.EDGE_EPS
                cut = (
                    (x1 <= eps and xoff > 0)
                    or (y1 <= eps and yoff > 0)
                    or (x2 >= sw - eps and xoff + sw < W)
                    or (y2 >= sh - eps and yoff + sh < H)
                )

                members = []
                if slice_masks is not None:
                    m = slice_masks[i]
                    if m.shape != (sh, sw):
                        m = cv2.resize(
                            m.astype(np.float32), (sw, sh), interpolation=cv2.INTER_NEAREST
                        )
                    members = [((m > 0.5).astype(np.uint8), xoff, yoff)]

                d = dict(
                    idx=len(dets),
                    box=[x1 + xoff, y1 + yoff, x2 + xoff, y2 + yoff],
                    score=float(conf[i]),
                    cls=pi,
                    cut=cut,
                    members=members,
                    area=0.0,
                )
                self._set_area(d)
                dets.append(d)

        # ---- 2. suppress duplicates and fragments (whole objects ranked first) ----
        order = sorted(dets, key=lambda d: (d["cut"], -d["score"]))
        kept: List[Dict[str, Any]] = []
        for d in order:
            drop = False
            for k in kept:
                if not self.cross_class_nms and k["cls"] != d["cls"]:
                    continue
                if d["cut"] and k["cut"]:
                    continue  # two fragments: united in step 3, never dropped here
                iou, ios, used_masks = self._overlap(d, k)
                if iou > self.nms_iou:                       # same object seen twice
                    drop = True
                elif d["cut"] and not k["cut"]:              # fragment of a whole object
                    if ios > (self.FRAG_MASK_IOS if used_masks else self.FRAG_BOX_IOS):
                        drop = True
                if drop:
                    break
            if not drop:
                kept.append(d)

        # ---- 3. merge fragments of the same long object across tile seams ----
        merged_any = True
        while merged_any:
            merged_any = False
            for i in range(len(kept)):
                a = kept[i]
                if not a["cut"]:
                    continue
                for j in range(i + 1, len(kept)):
                    b = kept[j]
                    if not b["cut"] or a["cls"] != b["cls"]:
                        continue
                    _, ios, used_masks = self._overlap(a, b)
                    thr = self.MERGE_MASK_IOS if used_masks else self.MERGE_BOX_IOS
                    if ios > thr:
                        a["box"] = [
                            min(a["box"][0], b["box"][0]),
                            min(a["box"][1], b["box"][1]),
                            max(a["box"][2], b["box"][2]),
                            max(a["box"][3], b["box"][3]),
                        ]
                        a["score"] = max(a["score"], b["score"])
                        a["idx"] = min(a["idx"], b["idx"])
                        a["members"] = (
                            a["members"] + b["members"]
                            if a["members"] and b["members"]
                            else []
                        )
                        self._set_area(a)
                        kept.pop(j)
                        merged_any = True
                        break
                if merged_any:
                    break

        kept.sort(key=lambda d: d["idx"])

        # ---- 4. paste kept masks onto full-size canvases ----
        masks: List[np.ndarray] = []
        if kept and all(d["members"] for d in kept):
            for d in kept:
                full = np.zeros((H, W), dtype=np.uint8)
                for m, xo, yo in d["members"]:
                    mh, mw = m.shape
                    h2, w2 = min(mh, H - yo), min(mw, W - xo)
                    full[yo : yo + h2, xo : xo + w2] |= m[:h2, :w2]
                masks.append(full)

        return {
            "boxes": [d["box"] for d in kept],
            "scores": [d["score"] for d in kept],
            "class_ids": [d["cls"] for d in kept],
            "class_names": class_names,
            "masks": masks,
            "debug": stats,
        }

    @staticmethod
    def mask_to_polygons(mask: np.ndarray) -> list:
        mask = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt.flatten().tolist() for cnt in contours if len(cnt) >= 3]

    # ------------------------------------------------------------------ #
    # Counting
    # ------------------------------------------------------------------ #

    @staticmethod
    def count_objects(labels: List[str], prompts: List[str]) -> Dict[str, Any]:
        """Per-class counts (zero-filled for every prompt) plus the grand total."""
        per_class = {p: 0 for p in prompts}
        for lbl in labels:
            per_class[lbl] = per_class.get(lbl, 0) + 1
        return {"per_class": per_class, "total": len(labels)}

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    @staticmethod
    def draw_boxes(image, boxes, scores, labels):
        img = image.copy()
        for b, s, lbl in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{lbl}:{s:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 4, y1)
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw, ty), (0, 255, 0), -1)
            cv2.putText(
                img, text, (x1, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        return img

    @staticmethod
    def draw_count_summary(image, counts):
        """Burn 'Total' and per-class counts into the image (top-left panel).

        Font size scales with image resolution so it stays readable on
        large photos; a semi-transparent dark panel keeps it legible on
        any background.
        """
        img = image.copy()
        H, W = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = max(0.6, min(H, W) / 1000.0 * 0.9)
        thick = max(1, int(round(scale * 2)))
        pad = int(12 * scale)

        lines = [f"Total objects: {counts['total']}"] + [
            f"{k}: {v}" for k, v in counts["per_class"].items()
        ]
        sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
        line_h = max(h for _, h in sizes) + pad
        panel_w = min(W, max(w for w, _ in sizes) + 2 * pad)
        panel_h = min(H, line_h * len(lines) + pad)

        # semi-transparent background panel
        roi = img[0:panel_h, 0:panel_w]
        dark = np.zeros_like(roi)
        img[0:panel_h, 0:panel_w] = cv2.addWeighted(roi, 0.4, dark, 0.6, 0)

        y = pad + sizes[0][1]
        for i, text in enumerate(lines):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)  # total in yellow
            cv2.putText(img, text, (pad, y), font, scale, color, thick, cv2.LINE_AA)
            y += line_h
        return img

    @staticmethod
    def draw_masks(image, masks):
        img = image.copy()
        rng = np.random.default_rng(0)  # fixed seed -> reproducible colours
        for m in masks:
            color = rng.integers(0, 255, 3).tolist()
            overlay = np.zeros_like(image)
            overlay[m > 0.5] = color
            img = cv2.addWeighted(img, 1.0, overlay, 0.5, 0)
        return img

    # ------------------------------------------------------------------ #
    # Main per-image logic
    # ------------------------------------------------------------------ #

    def process_image(
        self, image_path: str, prompts: List[str], out_dirs: Dict[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """Process one image. Returns the counts dict, or None if unreadable."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Could not read image: {image_path}")
            return None

        name = f"{Path(image_path).stem}_{self.make_tag(prompts)}"

        slice_outputs = self.run_prediction(image, image_path, prompts)
        merged = self.merge_slice_results(slice_outputs, image.shape, prompts)

        labels = [
            self.resolve_label(cid, merged["class_names"], prompts)
            for cid in merged["class_ids"]
        ]
        counts = self.count_objects(labels, prompts)

        if self.debug:
            print(f"\n[DEBUG] {Path(image_path).name}")
            for pr, st in merged["debug"].items():
                thr = self.prompt_conf.get(pr, self.conf)
                print(
                    f"  '{pr}': raw={st['raw']} (all tiles, incl. duplicates) | "
                    f"best score={st['max_conf']:.3f} | threshold={thr} | "
                    f"passed threshold={st['passed']} | final count={counts['per_class'].get(pr, 0)}"
                )

        # ---- box image (with count summary) ----
        bbox_img = self.draw_boxes(image, merged["boxes"], merged["scores"], labels)
        bbox_img = self.draw_count_summary(bbox_img, counts)
        cv2.imwrite(str(out_dirs["bbox"] / f"{name}.jpg"), bbox_img)

        # ---- mask overlay image (with count summary) ----
        mask_img = self.draw_masks(image, merged["masks"])
        mask_img = self.draw_count_summary(mask_img, counts)
        cv2.imwrite(str(out_dirs["mask"] / f"{name}.jpg"), mask_img)

        # ---- optional binary instance mask (pixel value = instance_id, 0 = background) ----
        if self.save_binary_masks and merged["masks"]:
            inst = np.zeros(image.shape[:2], dtype=np.uint16)
            for i, m in enumerate(merged["masks"]):
                inst[m > 0] = i + 1
            cv2.imwrite(str(out_dirs["binary"] / f"{name}_instances.png"), inst)

        # ---- JSON ----
        json_data = {
            "image": image_path,
            "prompts": prompts,
            "counts": counts,
            "detections": [],
        }
        for i, box in enumerate(merged["boxes"]):
            json_data["detections"].append(
                {
                    "instance_id": i + 1,
                    "bbox_xyxy": box,
                    "confidence": merged["scores"][i],
                    "class_id": merged["class_ids"][i],
                    "label": labels[i],
                    "segmentation": (
                        self.mask_to_polygons(merged["masks"][i])
                        if i < len(merged["masks"])
                        else []
                    ),
                }
            )
        with open(out_dirs["json"] / f"{name}.json", "w") as f:
            json.dump(json_data, f, indent=4)

        return counts


# ------------------------------------------------------------------ #
# CSV summary
# ------------------------------------------------------------------ #

def write_counts_csv(rows: List[Tuple[str, Dict[str, Any]]], csv_path: Path) -> None:
    labels = sorted({lbl for _, c in rows for lbl in c["per_class"]})
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", *labels, "total"])
        for img_name, c in rows:
            w.writerow([img_name, *[c["per_class"].get(l, 0) for l in labels], c["total"]])
        w.writerow(
            [
                "ALL_IMAGES",
                *[sum(c["per_class"].get(l, 0) for _, c in rows) for l in labels],
                sum(c["total"] for _, c in rows),
            ]
        )


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 text-prompt segmentation with optional SAHI slicing and counting"
    )
    parser.add_argument("--images", required=True, help="Directory of input images")
    parser.add_argument("--output", required=True, help="Directory for all outputs")
    parser.add_argument("--model", required=True, help="Path to SAM3 checkpoint (e.g. sam3.pt)")
    parser.add_argument("--text", nargs="+", required=True, help="Text prompt(s)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1008, help="Inference image size")
    parser.add_argument("--no-half", action="store_true", help="Use FP32 instead of FP16")
    parser.add_argument("--use-sahi", action="store_true", help="Enable slicing inference")
    parser.add_argument("--slice", type=int, default=1008, help="Slice size (SAHI)")
    parser.add_argument("--overlap", type=float, default=0.2, help="Slice overlap ratio (SAHI)")
    parser.add_argument("--nms-iou", type=float, default=0.35, help="Per-class NMS IoU threshold")
    parser.add_argument(
        "--prompt-conf",
        nargs="+",
        default=[],
        metavar="PROMPT=CONF",
        help='Per-prompt confidence override, e.g. --prompt-conf "nut=0.1" "screw=0.3"',
    )
    parser.add_argument("--debug", action="store_true", help="Print raw detections / best score per prompt")
    parser.add_argument(
        "--cross-class-nms",
        action="store_true",
        help="De-duplicate overlapping detections across different classes too",
    )
    parser.add_argument(
        "--save-binary-masks",
        action="store_true",
        help="Also save a uint16 instance-id mask PNG per image",
    )
    args = parser.parse_args()

    prompt_conf = {}
    for item in args.prompt_conf:
        key, val = item.rsplit("=", 1)
        prompt_conf[key.strip()] = float(val)

    proc = SAM3BatchProcessor(
        model_path=args.model,
        conf=args.conf,
        half=not args.no_half,
        imgsz=args.imgsz,
        use_sahi=args.use_sahi,
        slice_size=args.slice,
        overlap_ratio=args.overlap,
        nms_iou=args.nms_iou,
        cross_class_nms=args.cross_class_nms,
        prompt_conf=prompt_conf,
        debug=args.debug,
        save_binary_masks=args.save_binary_masks,
    )

    out_dirs = proc.setup_output_dirs(args.output)
    images = sorted(
        p for p in Path(args.images).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"No images found in {args.images}")
        return

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for img in tqdm(images):
        counts = proc.process_image(str(img), args.text, out_dirs)
        if counts is not None:
            rows.append((img.name, counts))

    if rows:
        csv_path = Path(args.output) / "counts_summary.csv"
        write_counts_csv(rows, csv_path)

        grand_total = sum(c["total"] for _, c in rows)
        print(f"\nProcessed {len(rows)} image(s) | total objects: {grand_total}")
        for label in sorted({l for _, c in rows for l in c["per_class"]}):
            print(f"  {label}: {sum(c['per_class'].get(label, 0) for _, c in rows)}")
        print(f"Counts CSV: {csv_path}")


if __name__ == "__main__":
    main()