"""Generate the figures and LaTeX tables used by report.tex.

Run from the repo root after both solutions have been run:
    python Om_Rode/report/make_figures.py
Needs data/, Om_Rode/{AI,Non_AI}/outputs/, Om_Rode/AI/labels_sam/ and the training run.
"""
import csv
import glob
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OM = os.path.join(ROOT, "Om_Rode")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
sys.path[:0] = [os.path.join(OM, "Non_AI"), os.path.join(OM, "AI")]
from autolabel_sam import draw, load_masks  # noqa: E402
from count_objects import (count_discs, disc_model, find_round_holes, foreground_mask,  # noqa: E402
                           load_resized, single_object_area, valid_components)
from prepare_dataset import TEST, VAL  # noqa: E402

os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#1f5f99", "#d9822b"


def src(name):
    return glob.glob(os.path.join(ROOT, "data", "**", name + ".jpg"), recursive=True)[0]


def save(name, img, long_side=1400, q=85):
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(FIG, name), img, [cv2.IMWRITE_JPEG_QUALITY, q])


def crop_frac(img, y0, y1, x0, x1):
    h, w = img.shape[:2]
    return img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def read_counts(p, key="count"):
    with open(p) as f:
        return {os.path.splitext(os.path.basename(r["image"]))[0]: int(r[key]) for r in csv.DictReader(f)}


def split(n):
    return "test" if n in TEST else "val" if n in VAL else "train"


# ---------------------------------------------------------------- dataset overview
scenes = [("img4", "Sparse screws (43)"), ("20240713_193659", "Few screws (10)"),
          ("20240713_193907", "Dense screws (193)"), ("20240713_194256", "Screws + discs (98)"),
          ("20240713_194621", "Nut pile (354)")]
tiles = []
for n, _ in scenes:
    im = cv2.imread(src(n))
    h, w = im.shape[:2]
    s = 420 / h
    tiles.append(cv2.resize(im, (int(w * s), 420), interpolation=cv2.INTER_AREA))
fig, axs = plt.subplots(1, 5, figsize=(7.2, 2.0), gridspec_kw={"width_ratios": [t.shape[1] for t in tiles]})
for ax, t, (_, title) in zip(axs, tiles, scenes):
    ax.imshow(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=7.5)
    ax.axis("off")
plt.tight_layout(pad=0.2)
plt.savefig(os.path.join(FIG, "dataset.jpg"), dpi=220)
plt.close()

# ---------------------------------------------------------------- Non-AI pipeline steps
n = "20240713_194200"
img = load_resized(src(n))
filled, norm, valid = foreground_mask(img)
box = (0.42, 0.66, 0.22, 0.70)
na_ov = cv2.imread(os.path.join(OM, "Non_AI", "outputs", n + ".jpg"))
steps = [("(a) input", img), ("(b) flattened illumination", cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)),
         ("(c) Otsu + hole filling", cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)), ("(d) instance masks", na_ov)]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0))
for ax, (t, im) in zip(axs, steps):
    ax.imshow(cv2.cvtColor(crop_frac(im, *box), cv2.COLOR_BGR2RGB), cmap="gray")
    ax.set_title(t, fontsize=8)
    ax.axis("off")
plt.tight_layout(pad=0.2)
plt.savefig(os.path.join(FIG, "nonai_steps.jpg"), dpi=220)
plt.close()

# ---------------------------------------------------------------- disc cores (Non-AI)
n_, labels, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
keep = valid_components(labels, stats, valid)
plain = np.array([stats[i, 4] for i in range(1, n_) if keep[i]])
single = single_object_area(plain)
dm = disc_model(labels, stats, keep, single)
dt = cv2.distanceTransform((filled > 0).astype(np.uint8), cv2.DIST_L2, 5)
core = dt >= 0.8 * dm[1]
vis = img.copy()
vis[filled > 0] = (0.5 * vis[filled > 0] + 0.5 * np.array([200, 160, 90])).astype(np.uint8)
vis[core] = (0, 0, 230)
dtv = cv2.applyColorMap(cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
dtv[filled == 0] = 255
box2 = (0.43, 0.60, 0.30, 0.68)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
axs[0].imshow(cv2.cvtColor(crop_frac(dtv, *box2), cv2.COLOR_BGR2RGB))
axs[0].set_title("(a) distance transform of the foreground", fontsize=8)
axs[1].imshow(cv2.cvtColor(crop_frac(vis, *box2), cv2.COLOR_BGR2RGB))
axs[1].set_title(f"(b) disc cores (red): distance $\\geq$ 0.8 r, r = {dm[1]:.1f} px", fontsize=8)
for ax in axs:
    ax.axis("off")
plt.tight_layout(pad=0.2)
plt.savefig(os.path.join(FIG, "nonai_discs.jpg"), dpi=220)
plt.close()

# ---------------------------------------------------------------- nut holes (Non-AI)
n = "20240713_194621"
img = load_resized(src(n))
_, norm, _ = foreground_mask(img)
_, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
holes, _ = find_round_holes(th)
hv = img.copy()
for hx, hy, c in holes:
    cv2.drawContours(hv, [c], -1, (0, 0, 255), 2)
box3 = (0.38, 0.62, 0.30, 0.70)
na_ov = cv2.imread(os.path.join(OM, "Non_AI", "outputs", n + ".jpg"))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
axs[0].imshow(cv2.cvtColor(crop_frac(hv, *box3), cv2.COLOR_BGR2RGB))
axs[0].set_title(f"(a) round holes found: {len(holes)} (one per nut)", fontsize=8)
axs[1].imshow(cv2.cvtColor(crop_frac(na_ov, *box3), cv2.COLOR_BGR2RGB))
axs[1].set_title("(b) nearest-hole (Voronoi) split into nut masks", fontsize=8)
for ax in axs:
    ax.axis("off")
plt.tight_layout(pad=0.2)
plt.savefig(os.path.join(FIG, "nonai_nuts.jpg"), dpi=220)
plt.close()

# ---------------------------------------------------------------- SAM auto vs corrected
LAB = os.path.join(OM, "AI", "labels_sam")
pairs = [("20240713_193907", 1144, 1837, "B14: screw merged with a disc"),
         ("20240713_194206", 857, 1156, "B16: two missed black screws")]
fig, axs = plt.subplots(2, 2, figsize=(5.6, 5.4))
for row, (n, x, y, cap) in zip(axs, pairs):
    im = cv2.imread(os.path.join(LAB, "images", n + ".jpg"))
    before, _ = load_masks(os.path.join(LAB, "masks_auto", n + ".npz"))
    after, _ = load_masks(os.path.join(LAB, "masks", n + ".npz"))
    r = 170
    for ax, ms, t in zip(row, (before, after), ("SAM automatic", "after one-click correction")):
        ov = draw(im, ms)
        ax.imshow(cv2.cvtColor(ov[y - r:y + r, x - r:x + r], cv2.COLOR_BGR2RGB))
        ax.set_title(f"{cap}\n{t} ({len(ms)} objects)", fontsize=7.5)
        ax.axis("off")
plt.tight_layout(pad=0.3)
plt.savefig(os.path.join(FIG, "sam_correction.jpg"), dpi=220)
plt.close()

# ---------------------------------------------------------------- full SAM overlay example
ov = cv2.imread(os.path.join(LAB, "overlays", "20240713_193831.jpg"))
save("sam_overlay.jpg", crop_frac(ov, 0.25, 0.75, 0.1, 0.9), 1600)

# ---------------------------------------------------------------- training curves
with open(os.path.join(OM, "AI", "runs", "yolov8s_seg", "results.csv")) as f:
    rr = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
ep = [r["epoch"] for r in rr]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
for k, lab in [("train/box_loss", "box"), ("train/seg_loss", "mask"), ("train/cls_loss", "cls")]:
    axs[0].plot(ep, [r[k] for r in rr], label=f"train {lab}", lw=1.2)
axs[0].plot(ep, [r["val/seg_loss"] for r in rr], label="val mask", lw=1.2, ls="--")
axs[0].set_xlabel("epoch"); axs[0].set_ylabel("loss"); axs[0].legend(fontsize=7, frameon=False)
axs[1].plot(ep, [r["metrics/mAP50(M)"] for r in rr], color=BLUE, lw=1.2, label="mask mAP50")
axs[1].plot(ep, [r["metrics/mAP50-95(M)"] for r in rr], color=ORANGE, lw=1.2, label="mask mAP50-95")
best = int(np.argmax([0.1 * r["metrics/mAP50(M)"] + 0.9 * r["metrics/mAP50-95(M)"] for r in rr]))
axs[1].axvline(ep[best], color="gray", lw=0.8, ls=":")
axs[1].text(ep[best] + 1, 0.95, f"best.pt (epoch {int(ep[best])})", fontsize=7, color="gray", va="top")
axs[1].set_xlabel("epoch"); axs[1].set_ylabel("validation mAP"); axs[1].set_ylim(0, 1)
axs[1].legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0))
plt.tight_layout(pad=0.3)
plt.savefig(os.path.join(FIG, "training.pdf"))
plt.close()
for f in ("val_batch0_pred.jpg",):
    im = cv2.imread(os.path.join(OM, "AI", "runs", "yolov8s_seg", f))
    if im is not None:
        save("val_pred.jpg", im, 1400)

# ---------------------------------------------------------------- accuracy comparison
gt = read_counts(os.path.join(OM, "ground_truth.csv"))
ai = read_counts(os.path.join(OM, "AI", "outputs", "counts.csv"))
na = read_counts(os.path.join(OM, "Non_AI", "outputs", "counts.csv"))
sam = {}
import json  # noqa: E402
notes = json.load(open(os.path.join(OM, "AI", "review_notes.json")))
for k in gt:
    sam[k] = notes.get(k, {}).get("auto", gt[k])
order = sorted(gt, key=lambda k: (not k.startswith("img"), k))
acc = lambda p, g: 100 * (1 - abs(p - g) / g)
fig, ax = plt.subplots(figsize=(7.2, 2.8))
x = np.arange(len(order))
ax.bar(x - 0.2, [acc(na[k], gt[k]) for k in order], 0.4, color=ORANGE, label="Non-AI (OpenCV)")
ax.bar(x + 0.2, [acc(ai[k], gt[k]) for k in order], 0.4, color=BLUE, label="AI (YOLOv8s-seg)")
ax.axhline(95, color="#b23a2e", lw=0.9, ls="--", label="95 % target")
ax.set_ylim(90, 100.6)
ax.set_xticks(x)
ax.set_xticklabels([k.replace("20240713_", "") + ("*" if split(k) == "test" else "") for k in order], rotation=70, fontsize=6.5)
ax.set_ylabel("count accuracy (%)")
ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
plt.tight_layout(pad=0.3)
plt.savefig(os.path.join(FIG, "accuracy.pdf"))
plt.close()

# ---------------------------------------------------------------- side-by-side outputs on test images
picks = [("20240713_193907", (0.35, 0.65, 0.2, 0.8)), ("20240713_194256", (0.38, 0.62, 0.22, 0.78)),
         ("20240713_194621", (0.38, 0.62, 0.25, 0.75)), ("img4", (0.15, 0.85, 0.2, 0.8))]
for n, b in picks:
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, d, t in zip(axs, ("Non_AI", "AI"), ("Non-AI", "AI")):
        im = cv2.imread(os.path.join(OM, d, "outputs", n + ".jpg"))
        ax.imshow(cv2.cvtColor(crop_frac(im, *b), cv2.COLOR_BGR2RGB))
        cnt = na[n] if d == "Non_AI" else ai[n]
        ax.set_title(f"{t}: {cnt} (ground truth {gt[n]})", fontsize=8)
        ax.axis("off")
    plt.tight_layout(pad=0.2)
    plt.savefig(os.path.join(FIG, f"compare_{n}.jpg"), dpi=200)
    plt.close()

# ---------------------------------------------------------------- tables
rows = []
for k in order:
    rows.append(f"{k.replace('_', chr(92) + '_')} & {split(k)} & {gt[k]} & {sam[k]} & {na[k]} & {acc(na[k], gt[k]):.1f} & {ai[k]} & {acc(ai[k], gt[k]):.1f} \\\\")
with open(os.path.join(TAB, "per_image.tex"), "w") as f:
    f.write("\n".join(rows) + "\n")
summ = []
for s in ("train", "val", "test", "all"):
    ks = [k for k in order if s == "all" or split(k) == s]
    summ.append(f"{s} & {len(ks)} & {np.mean([acc(na[k], gt[k]) for k in ks]):.2f} & {min(acc(na[k], gt[k]) for k in ks):.1f} & "
                f"{np.mean([acc(ai[k], gt[k]) for k in ks]):.2f} & {min(acc(ai[k], gt[k]) for k in ks):.1f} \\\\")
with open(os.path.join(TAB, "summary.tex"), "w") as f:
    f.write("\n".join(summ) + "\n")
corr = json.load(open(os.path.join(OM, "AI", "corrections.json")))
code = {k: f"{'S' if k.startswith('img') else 'B'}{i + 1:02d}" for i, k in enumerate(order)}
crow = []
for k, c in corr.items():
    crow.append(f"{code[k]} & {k.replace('_', chr(92) + '_')} & {len(c.get('add', []))} & {len(c.get('split', []))} & {sam[k]} & {gt[k]} \\\\")
with open(os.path.join(TAB, "corrections.tex"), "w") as f:
    f.write("\n".join(crow) + "\n")
print("figures:", sorted(os.listdir(FIG)))
print("tables :", sorted(os.listdir(TAB)))
