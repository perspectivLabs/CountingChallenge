"""Per-nut masks: watershed seeded at hole centres, limited to one nut radius."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import nut_holes as nh

def nut_masks(small):
    m = nh.foreground(small)
    res = nh.find_holes(m)
    if not res or not res[0]:
        return m, np.zeros(m.shape, np.int32), 0
    holes, _ = res
    filled = ndi.binary_fill_holes(m).astype(np.uint8)
    markers = np.zeros(m.shape, np.int32)
    for i, (_, _, c) in enumerate(holes, 1):
        markers[int(round(c[1])), int(round(c[0]))] = i
    # nut radius from isolated nuts: connected blobs that contain exactly one hole
    n, lab, st, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    per = {}
    for i, (_, _, c) in enumerate(holes, 1):
        l = lab[int(round(c[1])), int(round(c[0]))]
        per.setdefault(l, []).append(i)
    singles = [st[l, cv2.CC_STAT_AREA] for l, v in per.items() if l > 0 and len(v) == 1]
    area1 = np.median(singles) if len(singles) >= 3 else np.median([st[l, cv2.CC_STAT_AREA] / len(v) for l, v in per.items() if l > 0])
    R = np.sqrt(area1 / np.pi) * 1.08
    # nearest-hole (Voronoi) assignment, capped at one nut radius, inside the foreground
    seed = np.ones(m.shape, bool)
    for i, (_, _, c) in enumerate(holes, 1):
        seed[int(round(c[1])), int(round(c[0]))] = False
    d, idx = ndi.distance_transform_edt(seed, return_indices=True)
    nearest = markers[idx[0], idx[1]]
    out = np.where((filled > 0) & (d <= R), nearest, 0).astype(np.int32)
    return filled, out, len(holes)

def overlay(small, labels, count):
    rng = np.random.RandomState(3)
    cols = rng.randint(70, 255, (labels.max() + 1, 3)); cols[0] = 0
    blend = np.where(labels[..., None] > 0, (0.45 * small + 0.55 * cols[labels]).astype(np.uint8), small)
    edges = (cv2.morphologyEx((labels > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)
    cv2.putText(blend, f"nuts={count}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 255), 5)
    return blend

if __name__ == "__main__":
    for f in sys.argv[1:]:
        small = nh.load(f)
        _, lab, n = nut_masks(small)
        cov = (lab > 0).sum()
        print(os.path.basename(f), n, 'mask px', cov)
        cv2.imwrite(os.path.join(os.environ['OUT'], 'm_' + os.path.basename(f)), overlay(small, lab, n))
