"""Compare a counts.csv against ground_truth.csv.

Per-image accuracy = 1 - |predicted - true| / true. Images without a ground
truth entry are listed but not scored.

Usage:
  python evaluate.py Non_AI/output/counts.csv
  python evaluate.py AI/output/counts.csv --test-only
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# images never used to train the AI model (see AI/make_dataset.py)
TEST_IMAGES = {"img5", "20240713_193659", "20240713_194206", "20240713_194316", "20240713_194621"}


def read(path, col):
    with open(path, newline="") as f:
        return {r["image"]: int(r[col]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("counts")
    ap.add_argument("--gt", default=os.path.join(HERE, "ground_truth.csv"))
    ap.add_argument("--test-only", action="store_true")
    args = ap.parse_args()
    gt, pred = read(args.gt, "count"), read(args.counts, "count")

    accs, test_accs = [], []
    print(f"{'image':22s} {'pred':>5s} {'true':>5s} {'acc':>7s}")
    for name in sorted(pred):
        if args.test_only and name not in TEST_IMAGES:
            continue
        tag = " (test)" if name in TEST_IMAGES else ""
        if name not in gt:
            print(f"{name:22s} {pred[name]:5d}     -       -{tag}")
            continue
        acc = 1 - abs(pred[name] - gt[name]) / gt[name]
        accs.append(acc)
        if name in TEST_IMAGES:
            test_accs.append(acc)
        print(f"{name:22s} {pred[name]:5d} {gt[name]:5d} {acc:7.1%}{tag}")
    print(f"\nscored images: {len(accs)}   mean accuracy: {sum(accs) / len(accs):.1%}"
          f"   worst: {min(accs):.1%}")
    if test_accs:
        print(f"held-out test images: {len(test_accs)}   mean accuracy: "
              f"{sum(test_accs) / len(test_accs):.1%}")


if __name__ == "__main__":
    main()
