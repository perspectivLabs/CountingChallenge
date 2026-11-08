"""Compare predicted counts with the ground truth.

    python evaluate.py --pred Non_AI/outputs/counts.csv [AI/outputs/counts.csv ...] [--gt other_gt.csv]

Per-image counting accuracy = 1 - |pred - gt| / gt   (clipped at 0).
Reported: per-image table, mean accuracy, overall accuracy
(1 - sum|err| / sum gt), MAE and number of exact counts.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path: Path, key: str = "count") -> dict[str, int]:
    if not path.exists():
        raise SystemExit(f"file not found: {path.resolve()}")
    with open(path, newline="") as f:
        return {r["image"]: int(r[key]) for r in csv.DictReader(f)}


def evaluate(gt: dict[str, int], pred: dict[str, int]) -> tuple[list[dict], dict]:
    rows = []
    for img, g in gt.items():
        if img not in pred:
            continue
        p = pred[img]
        rows.append({"image": img, "gt": g, "pred": p, "error": p - g,
                     "accuracy": max(0.0, 1 - abs(p - g) / g)})
    if not rows:
        raise SystemExit("no overlapping images between prediction and ground truth")
    n = len(rows)
    summary = {
        "images": n,
        "mean_accuracy": sum(r["accuracy"] for r in rows) / n,
        "overall_accuracy": 1 - sum(abs(r["error"]) for r in rows) / sum(r["gt"] for r in rows),
        "min_accuracy": min(r["accuracy"] for r in rows),
        "mae": sum(abs(r["error"]) for r in rows) / n,
        "exact": sum(r["error"] == 0 for r in rows),
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True, help="counts.csv file(s) written by the solutions")
    ap.add_argument("--gt", default=str(HERE / "ground_truth.csv"))
    ap.add_argument("--markdown", help="optional path to write a markdown report")
    args = ap.parse_args()

    gt = load(Path(args.gt))
    md = []
    for pf in args.pred:
        rows, s = evaluate(gt, load(Path(pf)))
        print(f"\n=== {pf}")
        print(f"{'image':26s} {'gt':>5s} {'pred':>5s} {'err':>5s} {'acc':>7s}")
        for r in rows:
            print(f"{r['image']:26s} {r['gt']:5d} {r['pred']:5d} {r['error']:+5d} {100 * r['accuracy']:6.2f}%")
        print(f"mean accuracy {100 * s['mean_accuracy']:.2f}% | overall {100 * s['overall_accuracy']:.2f}% | "
              f"worst {100 * s['min_accuracy']:.2f}% | MAE {s['mae']:.2f} | exact {s['exact']}/{s['images']}")
        md.append(f"### {pf}\n\n| image | GT | pred | error | accuracy |\n|---|---:|---:|---:|---:|")
        md += [f"| {r['image']} | {r['gt']} | {r['pred']} | {r['error']:+d} | {100 * r['accuracy']:.2f}% |" for r in rows]
        md.append(f"\n**Mean accuracy {100 * s['mean_accuracy']:.2f}%**, overall {100 * s['overall_accuracy']:.2f}%, "
                  f"worst image {100 * s['min_accuracy']:.2f}%, MAE {s['mae']:.2f}, exact {s['exact']}/{s['images']}\n")
    if args.markdown:
        Path(args.markdown).write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
