"""Build a YOLO instance-segmentation dataset with exact masks via copy-paste synthesis.

Why synthetic data?
    No annotations are provided and hand-drawing polygons for ~4 000 touching
    or overlapping parts is impractical.  Instead:

    1. *Instance bank* - from every TRAIN image, cut out items that lie fully
       isolated (one item per connected component, not touching the border).
       For isolated items the classical segmentation is pixel-exact, so these
       cut-outs (RGB + alpha, nut holes transparent) are clean examples.
    2. *Backgrounds* - the same TRAIN images with all items inpainted away.
    3. *Scenes* - items are pasted with random rotation / scale / flip at
       random density, from sparse up to touching piles (nuts) and heavily
       overlapping heaps (screws).  Occlusion is tracked, so every pasted item
       gets an exact visible mask - even inside dense clusters, which is exactly
       where the classical method has to guess.
    4. *Real tiles* - crops of TRAIN images in which every item is isolated
       (their labels are exact), to keep the data close to the real domain.

HOLD-OUT images (at least one per object set) are never used to build the dataset,
so they measure generalisation to unseen photos.

    python synth_data.py --data <img folder> [<img folder> ...] --out dataset --n-train 1600 --n-val 200
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Non_AI"))
import count_non_ai as C  # noqa: E402  (classical segmentation is only used to cut out isolated items)

# at least one image of every object set is held out from data generation
HOLDOUT = {
    "20240713_193135",   # pan-head screws
    "20240713_193659",   # black countersunk screws
    "20240713_193907",   # dense small screws
    "20240713_194316",   # mixed screws + discs
    "20240713_194630",   # hex nuts
    "img5", "img6",      # brass screws (vignetted)
    "img1_43_nosy",      # noisy copy of img1 -> never trained on
}
TILE = 1024


@dataclass
class Item:
    rgba: np.ndarray   # h x w x 4, uint8
    area: int
    group: str         # source object set


def object_group(stem: str) -> str:
    """Object set of an image (items of one set share size/appearance)."""
    if stem.startswith("img"):
        return "brass"
    t = int(stem.split("_")[1])
    if t < 193500:
        return "panhead"
    if t < 193800:
        return "black"
    if t < 194000:
        return "small"
    if t < 194500:
        return "mixed"
    return "nut"


# --------------------------------------------------------------------------- #
# Instance bank + background extraction
# --------------------------------------------------------------------------- #
def analyse(img: np.ndarray):
    mask = C.segment(img)
    labels, st = C.split_instances(mask)
    filled, _ = C.find_holes(mask)
    return mask, filled, labels, st


def isolated_items(img, mask, filled, labels, st, group) -> tuple[list[Item], np.ndarray]:
    """Return isolated item cut-outs and a per-pixel map of 'exactly labelled' pixels."""
    H, W = mask.shape
    n, lab, cst, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    items, single_comp = [], np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, _ = cst[i]
        comp = lab[y:y + h, x:x + w] == i
        ids = np.unique(labels[y:y + h, x:x + w][comp])
        ids = ids[ids > 0]
        material = mask[y:y + h, x:x + w][comp].sum()
        ratio = material / max(st.unit_area, 1)
        if len(ids) != 1 or not (0.7 <= ratio <= 1.35):
            continue
        single_comp[i] = True
        if x == 0 or y == 0 or x + w == W or y + h == H:
            continue
        p = 3
        y0, y1, x0, x1 = max(0, y - p), min(H, y + h + p), max(0, x - p), min(W, x + w + p)
        alpha = ((lab[y0:y1, x0:x1] == i) & (mask[y0:y1, x0:x1] > 0)).astype(np.uint8) * 255
        rgba = np.dstack([img[y0:y1, x0:x1], alpha])
        items.append(Item(rgba, int((alpha > 0).sum()), group))
    exact = single_comp[lab]  # pixels whose instance label is trustworthy
    return items, exact


def clean_background(img: np.ndarray, filled: np.ndarray) -> np.ndarray:
    m = cv2.dilate(filled, C.disk(6))
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ms = cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    bg = cv2.inpaint(small, ms, 7, cv2.INPAINT_TELEA)
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = img.copy()
    out[m > 0] = bg[m > 0]
    return out


# --------------------------------------------------------------------------- #
# Scene synthesis
# --------------------------------------------------------------------------- #
def transform_item(it: Item, rng: random.Random) -> np.ndarray:
    rgba = it.rgba
    if rng.random() < 0.5:
        rgba = rgba[:, ::-1]
    s = rng.uniform(0.85, 1.15)
    ang = rng.uniform(0, 360)
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(np.ascontiguousarray(rgba), M, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # mild per-item photometric jitter
    rgb = out[..., :3].astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def compose(bg: np.ndarray, items: list[Item], rng: random.Random) -> tuple[np.ndarray, list[np.ndarray]]:
    H, W = bg.shape[:2]
    canvas = bg.astype(np.float32)
    inst = np.zeros((H, W), np.int32)
    area0: dict[int, int] = {}
    touching_only = items[0].group == "nut"          # real nuts lie flat and do not overlap
    mean_area = float(np.mean([i.area for i in items]))
    coverage = rng.choice([rng.uniform(0.02, 0.12), rng.uniform(0.12, 0.3), rng.uniform(0.3, 0.55)])
    n_target = int(coverage * H * W / mean_area)
    # objects cluster in piles in the real photos: sample around a few centres
    centres = [(rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.85) * H) for _ in range(rng.randint(1, 3))]
    spread = rng.uniform(0.15, 0.6) * min(H, W)
    diam = np.sqrt(mean_area) * 1.15
    p_adjacent = rng.choice([0.0, 0.5, 0.85])    # 0.85 ~ dense touching piles
    placed: list[tuple[float, float]] = []
    nid, tries = 0, 0
    while nid < n_target and tries < n_target * 10:
        tries += 1
        obj = transform_item(rng.choice(items), rng)
        oh, ow = obj.shape[:2]
        if placed and rng.random() < p_adjacent:
            # grow the pile: put the item right next to an already placed one
            px, py = rng.choice(placed)
            t = rng.uniform(0, 2 * np.pi)
            d = rng.uniform(0.95, 1.35) * diam
            x, y = int(px + d * np.cos(t) - ow / 2), int(py + d * np.sin(t) - oh / 2)
        elif rng.random() < 0.7:
            cx, cy = rng.choice(centres)
            x = int(rng.gauss(cx, spread) - ow / 2)
            y = int(rng.gauss(cy, spread) - oh / 2)
        else:
            x, y = rng.randint(-ow // 2, W - ow // 2), rng.randint(-oh // 2, H - oh // 2)
        xa, ya, xb, yb = max(0, x), max(0, y), min(W, x + ow), min(H, y + oh)
        if xb - xa < 4 or yb - ya < 4:
            continue
        a = obj[ya - y:yb - y, xa - x:xa - x + (xb - xa), 3].astype(np.float32) / 255.0
        hard = a > 0.5
        if hard.sum() < 0.5 * (obj[..., 3] > 127).sum():        # mostly outside the tile
            continue
        overlap = (inst[ya:yb, xa:xb][hard] > 0).mean()
        if touching_only and overlap > 0.03:
            continue
        if not touching_only and overlap > rng.choice([0.0, 0.25, 0.5]):
            continue
        a = cv2.GaussianBlur(a, (3, 3), 0.6)[..., None]          # soft edge like a real photo
        rgb = obj[ya - y:yb - y, xa - x:xa - x + (xb - xa), :3].astype(np.float32)
        canvas[ya:yb, xa:xb] = canvas[ya:yb, xa:xb] * (1 - a) + rgb * a
        nid += 1
        inst[ya:yb, xa:xb][hard] = nid
        area0[nid] = int(hard.sum())
        placed.append((x + ow / 2, y + oh / 2))
    # photometric post-processing of the whole scene
    canvas = canvas * rng.uniform(0.9, 1.1) + rng.uniform(-10, 10)
    canvas += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(0, 4), canvas.shape)
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.0))
    masks = []
    for j in range(1, nid + 1):
        m = inst == j
        if m.sum() >= 0.3 * area0[j] and m.sum() >= 25:   # drop items that are almost fully hidden
            masks.append(m)
    return img, masks


# --------------------------------------------------------------------------- #
# YOLO label writing
# --------------------------------------------------------------------------- #
def mask_to_polygon(m: np.ndarray) -> np.ndarray | None:
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 10 or len(c) < 3:
        return None
    c = cv2.approxPolyDP(c, 0.8, True)
    return c.reshape(-1, 2) if len(c) >= 3 else None


def write_sample(img: np.ndarray, masks: list[np.ndarray], img_path: Path, lbl_path: Path,
                 rng: random.Random) -> int:
    H, W = img.shape[:2]
    lines = []
    for m in masks:
        poly = mask_to_polygon(m)
        if poly is None:
            continue
        xy = (poly / [W, H]).clip(0, 1).ravel()
        lines.append("0 " + " ".join(f"{v:.5f}" for v in xy))
    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(75, 95)])
    lbl_path.write_text("\n".join(lines))
    return len(lines)


def real_tiles(img, labels, exact, rng: random.Random, n: int):
    """Tiles of real TRAIN images in which every labelled pixel is exact."""
    H, W = labels.shape
    out = []
    for _ in range(n * 10):
        if len(out) >= n:
            break
        th, tw = min(TILE, H), min(TILE, W)
        y, x = rng.randint(0, H - th), rng.randint(0, W - tw)
        lab = labels[y:y + th, x:x + tw]
        if (lab > 0).sum() == 0 or (exact[y:y + th, x:x + tw] | (lab == 0)).mean() < 0.9999:
            continue
        masks = [lab == j for j in np.unique(lab) if j > 0 and (lab == j).sum() >= 25]
        out.append((img[y:y + th, x:x + tw], masks))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="image folders / files")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--n-train", type=int, default=1600)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--real-per-image", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=1024,
                    help="tile size; use 512 together with train.py --imgsz 512 to keep full object resolution")
    args = ap.parse_args()
    global TILE
    TILE = args.tile
    rng = random.Random(args.seed)
    jpeg_rng = random.Random(args.seed + 1)   # separate stream: JPEG quality does not shift scene sampling
    np.random.seed(args.seed)

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    bank: dict[str, list[Item]] = {}
    bgs: list[np.ndarray] = []
    reals = []
    for p in C.list_images(args.data):
        if p.stem in HOLDOUT:
            print(f"[hold-out] {p.name}")
            continue
        img, _ = C.resize_long_side(cv2.imread(str(p)), C.WORK_SIDE)
        mask, filled, labels, st = analyse(img)
        g = object_group(p.stem)
        items, exact = isolated_items(img, mask, filled, labels, st, g)
        bank.setdefault(g, []).extend(items)
        bgs.append(clean_background(img, filled))
        reals += real_tiles(img, labels, exact, rng, args.real_per_image)
        print(f"[train]    {p.name:24s} group={g:8s} isolated items={len(items)}")
    print({g: len(v) for g, v in bank.items()}, "backgrounds", len(bgs), "real tiles", len(reals))

    groups = sorted(bank)
    stats = {"train": 0, "val": 0}
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        for k in range(n):
            bg = rng.choice(bgs)
            H, W = bg.shape[:2]
            y, x = rng.randint(0, H - min(TILE, H)), rng.randint(0, W - min(TILE, W))
            tile = bg[y:y + TILE, x:x + TILE]
            if rng.random() < 0.5:
                tile = tile[:, ::-1]
            if rng.random() < 0.5:
                tile = tile[::-1]
            g = rng.choice(groups)
            items = bank[g]
            # the mixed set always gets extra items from another set, the others
            # occasionally (10 %); nut scenes stay pure because nuts never overlap
            if g != "nut" and (g == "mixed" or rng.random() < 0.1):
                items = items + bank[rng.choice(groups)][:50]
            img, masks = compose(np.ascontiguousarray(tile), items, rng)
            stats[split] += write_sample(img, masks, out / "images" / split / f"syn_{k:05d}.jpg",
                                         out / "labels" / split / f"syn_{k:05d}.txt", jpeg_rng)
    for k, (img, masks) in enumerate(reals):
        stats["train"] += write_sample(img, masks, out / "images" / "train" / f"real_{k:05d}.jpg",
                                       out / "labels" / "train" / f"real_{k:05d}.txt", jpeg_rng)
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: item\n")
    print("instances written:", stats)


if __name__ == "__main__":
    main()
