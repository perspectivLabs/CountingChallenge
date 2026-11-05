"""Build ground truth from two independent counters plus a visual check of their disagreements.

Steps:
    prepare   Match every Non-AI instance with a SAM 2 instance one-to-one (Hungarian
              matching on mask IoU >= 0.3). Matched pairs are agreed objects. Unmatched
              instances are disputes; overlapping disputes from the two methods describe
              the same spot and are merged into one *candidate*. Each candidate is drawn
              on a zoomed review tile (``work/cand/``):
                  left   the plain image
                  right  thick red/blue outline = disputed mask (red Non-AI, blue SAM 2),
                         thin green outlines with dots = agreed objects
              Question per tile: is there a real object in the thick outline with no dot?
    review    Record the accepted candidates in ``review.json`` (by hand). A second pass over the
              medium-confidence images adds missed objects (``added``) and removes double
              points (``removed``, nearest point to the given location).
    finalize  Ground truth = agreed + accepted + added - removed -> ``../ground_truth.json``.

``ground_truth.json`` is the frozen reference. Candidate ids depend on the Non-AI version
that ``prepare`` ran with, so ``review.json`` also stores the point of every accepted
candidate.

Usage:
    uv run compare.py prepare  /path/to/data
    uv run compare.py finalize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Non_AI"))
import count as nonai  # noqa: E402  (Non-AI counter, a sibling project)

WORK = HERE / "work"
GT_FILE = HERE.parent / "ground_truth.json"
REVIEW = HERE / "review.json"
SIDE, ZOOM, PER_SHEET = 150, 2, 12  # review tile size, zoom factor, tiles per sheet


def inner_point(mask: np.ndarray) -> list[float]:
    """Return the mask pixel closest to the mask centroid (always on the object)."""
    ys, xs = np.nonzero(mask)
    j = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
    return [float(xs[j]), float(ys[j])]


def match(
    a: np.ndarray, b: np.ndarray, thr: float = 0.3
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match the instances of two label images one-to-one by IoU.

    Returns:
        Matched ``(id_a, id_b)`` pairs with IoU >= ``thr``, unmatched ids of ``a`` and of ``b``.
    """
    na, nb = a.max(), b.max()
    both = (a > 0) | (b > 0)
    keys, inter = np.unique(a[both].astype(np.int64) * (nb + 1) + b[both], return_counts=True)
    ia, ib = keys // (nb + 1), keys % (nb + 1)
    area_a = np.bincount(a.ravel(), minlength=na + 1)
    area_b = np.bincount(b.ravel(), minlength=nb + 1)
    sel = (ia > 0) & (ib > 0)
    iou = np.zeros((na + 1, nb + 1))
    iou[ia[sel], ib[sel]] = inter[sel] / (area_a[ia[sel]] + area_b[ib[sel]] - inter[sel])
    rows, cols = linear_sum_assignment(-iou[1:, 1:])
    pairs = [(i + 1, j + 1) for i, j in zip(rows, cols, strict=True) if iou[i + 1, j + 1] >= thr]
    matched_a, matched_b = {p[0] for p in pairs}, {p[1] for p in pairs}
    only_a = [i for i in range(1, na + 1) if area_a[i] and i not in matched_a]
    only_b = [j for j in range(1, nb + 1) if area_b[j] and j not in matched_b]
    return pairs, only_a, only_b


def candidates(disputes: list[tuple[str, int, np.ndarray]]) -> list[list[int]]:
    """Group disputed masks that overlap (>20% of the smaller one) into candidates."""
    groups: list[list[int]] = []
    for k, (_, _, m) in enumerate(disputes):
        hit = [
            g
            for g in groups
            if any((m & disputes[o][2]).sum() > 0.2 * min(m.sum(), disputes[o][2].sum()) for o in g)
        ]
        for g in hit:
            groups.remove(g)
        groups.append([*sum(hit, []), k])
    return groups


def review_tile(img, agreed, disputes, group, box) -> np.ndarray:
    """Draw one candidate: plain crop on the left, annotated crop on the right."""
    h, w = img.shape[:2]
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    x0 = int(min(max(cx - SIDE // 2, 0), w - SIDE))
    y0 = int(min(max(cy - SIDE // 2, 0), h - SIDE))
    sl = (slice(y0, y0 + SIDE), slice(x0, x0 + SIDE))

    def zoom(a, interp=cv2.INTER_NEAREST):
        return cv2.resize(a, None, fx=ZOOM, fy=ZOOM, interpolation=interp)

    plain = zoom(img[sl], cv2.INTER_CUBIC)
    ann = plain.copy()
    ag = zoom(agreed[sl].astype(np.uint16))
    for v in np.unique(ag)[1:]:
        m = (ag == v).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ann, contours, -1, (0, 200, 0), 1)
        x, y = inner_point(m > 0)
        cv2.circle(ann, (int(x), int(y)), 5, (0, 230, 0), -1)
    for k in group:
        src, _, m = disputes[k]
        contours, _ = cv2.findContours(zoom(m[sl].astype(np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ann, contours, -1, (0, 0, 255) if src == "N" else (255, 90, 0), 3)
    tile = np.hstack([plain, np.full((SIDE * ZOOM, 4, 3), 255, np.uint8), ann])
    return cv2.copyMakeBorder(tile, 30, 6, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def prepare(data: Path) -> None:
    """Match both counters on every image and write agreed points, candidates and review tiles."""
    (WORK / "cand").mkdir(parents=True, exist_ok=True)
    for f in sorted(data.rglob("*.jpg")):
        sam_file = HERE / "sam" / f"{f.parent.name}__{f.stem}.npz"
        if not sam_file.exists():
            print(f"no SAM 2 result for {f.name}; run sam_count.py first")
            continue
        res = nonai.count_image(f)
        a, b = res.labels, np.load(sam_file)["labels"]
        pairs, only_a, only_b = match(a, b)
        disputes = [("N", i, a == i) for i in only_a] + [("S", j, b == j) for j in only_b]
        groups = candidates(disputes)

        agreed = np.zeros(a.shape, np.int32)
        for n, (i, _) in enumerate(pairs, 1):
            agreed[a == i] = n
        cands, tiles = [], []
        for c, group in enumerate(groups):
            union = np.any([disputes[k][2] for k in group], axis=0)
            ys, xs = np.nonzero(union)
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            largest = max(group, key=lambda k: disputes[k][2].sum())
            members = [f"{disputes[k][0]}{disputes[k][1]}" for k in group]
            cands.append(
                {"id": f"C{c}", "members": members, "point": inner_point(disputes[largest][2]), "box": box}
            )
            tile = review_tile(res.image, agreed, disputes, group, box)
            cv2.putText(
                tile, f"C{c}  ({' '.join(members)})", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2
            )
            tiles.append(tile)

        for s in range(0, len(tiles), PER_SHEET):
            chunk = tiles[s : s + PER_SHEET]
            chunk += [np.full_like(tiles[0], 255)] * (-len(chunk) % 3)
            sheet = np.vstack([np.hstack(chunk[r : r + 3]) for r in range(0, len(chunk), 3)])
            cv2.imwrite(
                str(WORK / "cand" / f"{f.stem}_{s // PER_SHEET}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

        key = f"{f.parent.name}/{f.name}"
        info = {"key": key, "agreed": [inner_point(a == i) for i, _ in pairs], "candidates": cands}
        (WORK / f"{f.parent.name}__{f.stem}.json").write_text(json.dumps(info))
        print(f"{key:42s} agreed {len(pairs):4d}  disputes {len(disputes):3d}  candidates {len(cands):3d}")


def finalize() -> None:
    """Write ``../ground_truth.json`` from the agreed points and the review decisions."""
    review = json.loads(REVIEW.read_text())
    gt = {}
    for wf in sorted(WORK.glob("*.json")):
        info = json.loads(wf.read_text())
        key = info["key"]
        if key not in review:
            print(f"{key}: not reviewed yet, skipped")
            continue
        accepted = [c["point"] for c in review[key]["accepted"]]
        points = info["agreed"] + accepted + review[key].get("added", [])
        for x, y in review[key].get("removed", []):
            nearest = min(range(len(points)), key=lambda k: (points[k][0] - x) ** 2 + (points[k][1] - y) ** 2)
            points.pop(nearest)
        gt[key] = {
            "count": len(points),
            "checked_by": "Non-AI and SAM 2 agree; disagreements checked visually",
            "confidence": review[key]["confidence"],
            "points": [[round(x, 1), round(y, 1)] for x, y in points],
        }
        print(f"{key:42s} agreed {len(info['agreed']):4d} + accepted {len(accepted):3d} -> {len(points)}")
    GT_FILE.write_text(json.dumps(gt, indent=1))


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description="Build ground truth from Non-AI and SAM 2 agreement.")
    ap.add_argument("step", choices=["prepare", "finalize"])
    ap.add_argument("data", type=Path, nargs="?", help="dataset folder (prepare only)")
    args = ap.parse_args()
    if args.step == "prepare":
        if args.data is None:
            ap.error("prepare needs the dataset folder")
        prepare(args.data)
    else:
        finalize()


if __name__ == "__main__":
    main()
