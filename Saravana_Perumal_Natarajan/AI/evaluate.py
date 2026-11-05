"""Score the AI counter against ``../ground_truth.json``, with the held-out test images reported separately.

Same metrics as ``../Non_AI/evaluate.py``: count accuracy = 1 - |predicted - true| / true,
plus per-point mask checks (matched / merged / missed / false positive).

Usage:
    uv run evaluate.py /path/to/data -o outputs [--weights PATH] [--conf 0.35]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter as Tally
from pathlib import Path

import cv2
import numpy as np

from count import DEFAULT_WEIGHTS, Counter, overlay
from splits import split_of

GT_FILE = Path(__file__).resolve().parent.parent / "ground_truth.json"


def score(labels: np.ndarray, count: int, points: list[list[float]]) -> dict:
    """Compare an instance label image with ground-truth points (one per object)."""
    h, w = labels.shape
    hits: Tally[int] = Tally()
    missed = 0
    for x, y in points:
        xi = min(max(round(x), 0), w - 1)
        yi = min(max(round(y), 0), h - 1)
        window = labels[max(yi - 3, 0) : yi + 4, max(xi - 3, 0) : xi + 4]  # tolerate points just off the edge
        ids = window[window > 0]
        if len(ids):
            hits[int(np.bincount(ids).argmax())] += 1
        else:
            missed += 1
    true = len(points)
    return {
        "true": true,
        "pred": count,
        "count_acc": 1 - abs(count - true) / true,
        "matched": len(hits),
        "merged": sum(v - 1 for v in hits.values()),
        "missed": missed,
        "false_pos": count - len(hits),
    }


def summarise(rows: dict[str, dict], title: str) -> None:
    """Print mean / worst per-image accuracy and total-count accuracy."""
    if not rows:
        return
    accs = [r["count_acc"] for r in rows.values()]
    true = sum(r["true"] for r in rows.values())
    pred = sum(r["pred"] for r in rows.values())
    print(
        f"{title}: mean per-image accuracy {np.mean(accs):.2%} (min {min(accs):.2%}), "
        f"total true {true} predicted {pred} -> {1 - abs(pred - true) / true:.2%}"
    )


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description="Evaluate the AI counter.")
    ap.add_argument("data", help="dataset folder")
    ap.add_argument("-o", "--out", default="outputs", help="folder for overlays and results.json")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    gt = json.loads(GT_FILE.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counter = Counter(args.weights, args.conf)
    rows = {}
    header = f"{'image':42s} {'split':5s} {'true':>5s} {'pred':>5s} {'acc':>7s}"
    print(f"{header} {'match':>6s} {'merge':>6s} {'miss':>5s} {'fp':>4s}")
    for f in sorted(Path(args.data).rglob("*.jpg")):
        key = f"{f.parent.name}/{f.name}"
        if key not in gt:
            continue
        res = counter(f)
        r = rows[key] = {**score(res.labels, res.count, gt[key]["points"]), "split": split_of(f.name)}
        cv2.imwrite(str(out / f"{f.parent.name}__{f.stem}.jpg"), overlay(res), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(
            f"{key:42s} {r['split']:5s} {r['true']:5d} {r['pred']:5d} {r['count_acc']:7.2%} "
            f"{r['matched']:6d} {r['merged']:6d} {r['missed']:5d} {r['false_pos']:4d}",
            flush=True,
        )
    print()
    for split, title in (("test", "TEST (held out)"), ("val", "VAL"), ("train", "TRAIN")):
        summarise({k: v for k, v in rows.items() if v["split"] == split}, title)
    summarise(rows, "ALL")
    (out / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
