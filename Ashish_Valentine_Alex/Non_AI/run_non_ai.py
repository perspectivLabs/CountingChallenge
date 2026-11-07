"""Non-AI object counter: counts items in every image and writes mask overlays.

Usage:
    python run_non_ai.py --input <dataset_dir> --output <out_dir>

Two classical pipelines, chosen automatically per image:
  * nut scenes   - every hex nut has one round see-through hole; holes are detected, counted, and each nut's
                   mask is the foreground pixels nearest to its hole (capped at one nut radius).
  * screw scenes - background subtraction + threshold + morphology + connected components, with an
                   area-guided distance-transform watershed to split touching screws.
Outputs per image: <name>.jpg overlay (one colour per object, count printed), <name>_labels.png (16-bit
instance-label map at working resolution) and a counts.csv with the scene type.
"""
import argparse, csv, glob, os
import cv2
import numpy as np

import count_classical as screws
import nut_holes
import nut_masks

NUT_SCENE_MIN_HOLES = 100  # nut scenes have ~350 holes; every other scene has < 40


def count_image(path):
    img = cv2.imread(path)
    small_nut = nut_holes.load(path)
    filled, labels, n_nuts = nut_masks.nut_masks(small_nut)
    if n_nuts >= NUT_SCENE_MIN_HOLES:
        return "nuts", small_nut, labels, n_nuts
    small, mask = screws.segment(img)
    labels, n = screws.split_and_count(mask)
    return "screws", small, labels, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.input, "**", "*.jpg"), recursive=True)):
        scene, small, labels, n = count_image(f)
        name = os.path.splitext(os.path.relpath(f, a.input).replace(os.sep, "_"))[0]
        ov = screws.overlay(small, labels, n)
        cv2.imwrite(os.path.join(a.output, name + ".jpg"), ov)
        cv2.imwrite(os.path.join(a.output, name + "_labels.png"), labels.astype(np.uint16))
        rows.append((os.path.relpath(f, a.input), scene, n, small.shape[1], small.shape[0]))
        print(rows[-1])
    with open(os.path.join(a.output, "counts.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "scene", "count", "work_width", "work_height"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
