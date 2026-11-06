"""Non-AI screw counter (OpenCV only).
Pipeline: grayscale -> background flatten (divide by large blur) -> Otsu threshold
-> morphology -> connected components -> area-based split of touching screws.
Usage: python count_nonai.py <image_dir> <out_dir>
"""
import sys, glob, os, csv
import cv2
import numpy as np

MAX_SIDE = 1600  # work at reduced resolution for speed/consistency


def segment(gray):
    k = int(max(gray.shape) / 8) | 1
    bg = cv2.medianBlur(cv2.resize(gray, None, fx=0.1, fy=0.1), 31)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]))
    bg = cv2.GaussianBlur(bg, (k, k), 0)
    norm = cv2.divide(gray, bg, scale=255)            # flatten uneven lighting
    norm = cv2.GaussianBlur(norm, (3, 3), 0)
    _, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    # absolute darkness gate: removes vignette/shadow blobs that only look dark after flattening
    dark = gray < (np.percentile(gray, 50) * 0.75)
    mask = cv2.bitwise_and(mask, dark.astype(np.uint8) * 255)
    return mask


def count_image(img):
    s = MAX_SIDE / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = segment(gray)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return img, mask, lab, 0, []
    min_area = max(30, 0.2 * np.median(areas))        # drop dust specks
    H, W = gray.shape
    typical = np.median(areas[areas >= min_area]) if (areas >= min_area).any() else 1
    keep = []
    for i, a in enumerate(areas):
        i += 1
        if a < min_area:
            continue
        x, y, w, h = stats[i, :4]
        touches = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        # big blobs glued to the image border are lens vignette, not objects
        if touches and a > 2.0 * typical:
            continue
        cx, cy = x + w / 2, y + h / 2
        in_corner = (cx < 0.08 * W or cx > 0.92 * W) and (cy < 0.12 * H or cy > 0.88 * H)
        if touches and in_corner:       # lens-vignette slivers in the image corners
            continue
        if touches and gray[lab == i].mean() > 0.6 * np.median(gray):
            continue
        keep.append(i)
    kept_areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in keep])
    # single-screw reference area: median of blobs close to the lower half of sizes
    single = np.median(kept_areas[kept_areas <= np.percentile(kept_areas, 60)])
    per_blob = []
    total = 0
    for i, a in zip(keep, kept_areas):
        c = max(1, int(round(a / single))) if a > 1.6 * single else 1
        per_blob.append((i, c))
        total += c
    return img, mask, lab, total, per_blob


def overlay(img, lab, per_blob, total):
    out = img.copy()
    rng = np.random.RandomState(1)
    col = out.copy()
    for i, c in per_blob:
        color = tuple(int(x) for x in rng.randint(60, 255, 3))
        col[lab == i] = color
    out = cv2.addWeighted(col, 0.6, out, 0.4, 0)
    for i, c in per_blob:
        ys, xs = np.where(lab == i)
        cv2.putText(out, str(c) if c > 1 else "", (int(xs.mean()), int(ys.min()) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(out, f"Count: {total}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    return out


def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, "*.jpg")) + glob.glob(os.path.join(in_dir, "*.png")))
    rows = []
    for f in files:
        img = cv2.imread(f)
        img, mask, lab, total, pb = count_image(img)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), overlay(img, lab, pb, total))
        rows.append((os.path.basename(f), total))
        print(os.path.basename(f), total)
    with open(os.path.join(out_dir, "counts.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows([("image", "count")] + rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])