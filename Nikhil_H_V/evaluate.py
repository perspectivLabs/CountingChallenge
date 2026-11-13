"""Compare predicted counts with ground_truth.csv.

Accuracy per image = 1 - |pred - gt| / gt ; overall = mean over images.
Also reports the total-count accuracy (1 - sum|err| / sum gt).

Usage (from this folder):
  python evaluate.py Non_AI/outputs/counts.csv AI/outputs/counts.csv
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    with open(path, newline="") as fh:
        return {r["image"]: int(r["count"]) for r in csv.DictReader(fh)}


def main(paths):
    gt = load(os.path.join(HERE, "ground_truth.csv"))
    preds = {os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p)))) or p: load(p) for p in paths}
    names = list(preds)
    print(f"{'image':45s} {'GT':>5s} " + " ".join(f"{n:>14s}" for n in names))
    acc = {n: [] for n in names}
    abs_err = {n: 0 for n in names}
    for img, g in gt.items():
        cells = []
        for n in names:
            p = preds[n].get(img)
            if p is None:
                cells.append(f"{'-':>14s}")
                continue
            a = 1 - abs(p - g) / g
            acc[n].append(a)
            abs_err[n] += abs(p - g)
            cells.append(f"{p:>6d} ({a*100:5.1f}%)")
        print(f"{img:45s} {g:5d} " + " ".join(cells))
    total = sum(gt.values())
    print()
    for n in names:
        mean = sum(acc[n]) / len(acc[n])
        print(f"{n:8s} mean per-image accuracy = {mean*100:.2f}%   "
              f"total-count accuracy = {(1 - abs_err[n]/total)*100:.2f}%   "
              f"worst image = {min(acc[n])*100:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:] or [os.path.join(HERE, "Non_AI", "outputs", "counts.csv"),
                          os.path.join(HERE, "AI", "outputs", "counts.csv")])
