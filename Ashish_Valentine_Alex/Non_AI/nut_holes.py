"""Nut-hole detector: every hex nut has one round see-through hole -> count holes."""
import sys, glob, os
import cv2, numpy as np
from scipy import ndimage as ndi

LONG = 2400

def load(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = LONG / max(h, w)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def background(gray):
    """Bright-surface estimate that survives large dark piles: max filter on a small copy, then smooth."""
    sm = cv2.resize(gray, (gray.shape[1]//8, gray.shape[0]//8), interpolation=cv2.INTER_AREA)
    k = int(max(sm.shape) * 0.5) | 1
    bg = cv2.dilate(sm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), max(sm.shape) * 0.15)
    return cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)

def foreground(small, ratio=0.72):
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    bg = background(gray)
    r = gray.astype(np.float32) / np.maximum(bg, 1)
    return (r < ratio).astype(np.uint8)

def find_holes(mask, min_circ=0.85):
    filled = ndi.binary_fill_holes(mask).astype(np.uint8)
    enclosed = (filled > 0) & (mask == 0)
    n, lab, st, cen = cv2.connectedComponentsWithStats(enclosed.astype(np.uint8), connectivity=4)
    cand = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 12: continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        p = cv2.arcLength(cnts[0], True)
        circ = 4*np.pi*a/(p*p+1e-9)
        cand.append((a, circ, cen[i]))
    if not cand: return []
    areas = np.array([c[0] for c in cand]); circs = np.array([c[1] for c in cand])
    good = areas[circs >= min_circ]
    ref = np.median(good) if len(good) else np.median(areas)
    for _ in range(3):
        sel = good[(good > 0.5*ref) & (good < 1.8*ref)]
        if len(sel): ref = np.median(sel)
    return [(a, c, cen) for a, c, cen in cand if c >= min_circ and 0.4*ref <= a <= 2.2*ref], ref

if __name__ == "__main__":
    for f in sys.argv[1:]:
        small = load(f); m = foreground(small)
        holes, ref = find_holes(m)
        print(os.path.basename(f), 'holes=', len(holes), 'ref_area=%.0f' % ref)
        vis = small.copy()
        for a, c, cen in holes:
            cv2.circle(vis, (int(cen[0]), int(cen[1])), 6, (0, 0, 255), 2)
        cv2.putText(vis, f"holes={len(holes)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0,0,255), 5)
        cv2.imwrite(os.path.join(os.environ['OUT'], os.path.basename(f)), vis)
