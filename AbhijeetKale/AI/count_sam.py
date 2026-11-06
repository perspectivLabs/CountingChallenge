"""AI screw counter using Meta's Segment Anything (SAM) automatic mask generator.
Run on a GPU (Colab/Kaggle) -- see README. Usage:
  python count_sam.py <image_dir> <out_dir> [checkpoint.pth]
"""
import sys, os, glob, csv, urllib.request
import cv2, numpy as np, torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MAX_SIDE = 1600


def load(ckpt):
    if not os.path.exists(ckpt):
        urllib.request.urlretrieve(CKPT_URL, ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=ckpt).to(dev)
    return SamAutomaticMaskGenerator(sam, points_per_side=48, pred_iou_thresh=0.88,
                                     stability_score_thresh=0.92, min_mask_region_area=40)


def count(gen, img_bgr):
    s = MAX_SIDE / max(img_bgr.shape[:2])
    img_bgr = cv2.resize(img_bgr, None, fx=s, fy=s) if s < 1 else img_bgr
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    masks = gen.generate(rgb)
    H, W = gray.shape
    bg = np.median(gray)
    cand = []
    for m in masks:
        seg, a = m["segmentation"], m["area"]
        if a > 0.02 * H * W:                      # background / paper
            continue
        if gray[seg].mean() > 0.8 * bg:           # object must be darker than the backdrop
            continue
        cand.append(m)
    if not cand:
        return img_bgr, [], 0
    areas = np.array([m["area"] for m in cand])
    lo = np.median(areas) * 0.3
    cand = [m for m in cand if m["area"] >= lo]   # drop specks / sub-parts (heads, threads)
    cand.sort(key=lambda m: -m["area"])
    kept, occ = [], np.zeros((H, W), bool)         # NMS: skip masks mostly covered by a kept one
    for m in cand:
        seg = m["segmentation"]
        if (seg & occ).sum() > 0.5 * seg.sum():
            continue
        kept.append(m); occ |= seg
    return img_bgr, kept, len(kept)


def main(in_dir, out_dir, ckpt="sam_vit_b_01ec64.pth"):
    os.makedirs(out_dir, exist_ok=True)
    gen = load(ckpt)
    rows = []
    for f in sorted(glob.glob(os.path.join(in_dir, "*.jpg")) + glob.glob(os.path.join(in_dir, "*.png"))):
        img, masks, n = count(gen, cv2.imread(f))
        ov, rng = img.copy(), np.random.RandomState(0)
        for m in masks:
            ov[m["segmentation"]] = rng.randint(60, 255, 3)
        out = cv2.addWeighted(ov, 0.6, img, 0.4, 0)
        cv2.putText(out, f"Count: {n}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), out)
        rows.append((os.path.basename(f), n)); print(rows[-1])
    csv.writer(open(os.path.join(out_dir, "counts.csv"), "w", newline="")).writerows([("image", "count")] + rows)


if __name__ == "__main__":
    main(*sys.argv[1:])
