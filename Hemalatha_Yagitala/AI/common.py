"""Shared helpers for the AI pipeline."""
import cv2
import numpy as np

LONG_SIDE = 2000   # working resolution, same as the Non-AI pipeline (keeps labels aligned)
TILE = 640
OVERLAP = 160

# Held-out test images: one from every photo group, never used for training.
TEST_IMAGES = {
    "img2.jpg", "img5.jpg",
    "20240713_193135.jpg", "20240713_193659.jpg", "20240713_193907.jpg",
    "20240713_194232.jpg", "20240713_194316.jpg", "20240713_194621.jpg",
}

def load(path, long_side=LONG_SIDE):
    im = cv2.imread(path)
    s = min(1.0, long_side / max(im.shape[:2]))
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

def tiles(h, w, tile=TILE, overlap=OVERLAP):
    """Yield (y0, x0) tile origins covering the image, last tile flush with the edge."""
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < h: ys.append(h - tile)
    if xs[-1] + tile < w: xs.append(w - tile)
    for y in ys:
        for x in xs:
            yield max(y, 0), max(x, 0)

def overlay(im, masks, text):
    out = im.copy()
    rng = np.random.default_rng(7)
    for m in masks:
        c = rng.integers(40, 255, 3)
        out[m] = (0.45 * out[m] + 0.55 * c).astype(np.uint8)
    fs = max(im.shape[:2]) / 2000
    for k, m in enumerate(masks, 1):
        ys, xs = np.nonzero(m)
        if len(xs):
            cv2.putText(out, str(k), (int(xs.mean()) - 6, int(ys.mean()) + 4), 0, 0.33 * fs,
                        (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (int(330 * fs), int(55 * fs)), (0, 0, 0), -1)
    cv2.putText(out, text, (int(10 * fs), int(40 * fs)), 0, 1.2 * fs, (0, 255, 255), 2, cv2.LINE_AA)
    return out
