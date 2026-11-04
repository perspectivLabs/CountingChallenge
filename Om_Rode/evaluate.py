"""Compare predicted counts with ground_truth.csv.

Per-image accuracy = 1 - |pred - gt| / gt ; reported per image, per split and overall.

  python Om_Rode/evaluate.py Om_Rode/Non_AI/outputs/counts.csv
  python Om_Rode/evaluate.py Om_Rode/AI/outputs/counts.csv
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "AI"))
from prepare_dataset import TEST, VAL  # noqa: E402


def load(path, col):
    with open(path) as f:
        return {os.path.splitext(os.path.basename(r["image"]))[0]: int(r[col]) for r in csv.DictReader(f)}


def main():
    gt = load(os.path.join(HERE, "ground_truth.csv"), "count")
    pred = load(sys.argv[1], "count")
    rows, accs = [], {"train": [], "val": [], "test": []}
    for name in sorted(gt):
        if name not in pred:
            continue
        a = max(0.0, 1 - abs(pred[name] - gt[name]) / gt[name])
        sp = "test" if name in TEST else "val" if name in VAL else "train"
        accs[sp].append(a)
        rows.append((name, sp, gt[name], pred[name], a))
    print(f"{'image':<22}{'split':<7}{'gt':>5}{'pred':>6}{'acc':>8}")
    for n, sp, g, p, a in rows:
        print(f"{n:<22}{sp:<7}{g:>5}{p:>6}{a:>8.1%}")
    allv = [a for v in accs.values() for a in v]
    for sp, v in accs.items():
        if v:
            print(f"mean accuracy [{sp}]: {sum(v) / len(v):.2%}")
    print(f"mean accuracy [all]: {sum(allv) / len(allv):.2%}")


if __name__ == "__main__":
    main()
