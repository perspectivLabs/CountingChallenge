# Counting Challenge: Nikhil H V

The task is to count the items (screws, bolts and nuts) in each image and overlay a mask on every item. There are two solutions:

| Folder | Method | Mean per-image accuracy | Total-count accuracy |
|---|---|---|---|
| [`Non_AI/`](Non_AI) | Classical OpenCV: illumination flattening, Otsu threshold, hole/disc detection, area-based splitting, watershed | **97.75 %** | 97.72 % |
| [`AI/`](AI) | FastSAM (zero-shot "segment anything" model) on overlapping tiles, plus mask-hierarchy resolution | **98.68 %** | 99.09 % |

Accuracy per image is `1 - |predicted - true| / true`, measured on all 25 images in the dataset. The full per-image table is in [`results_summary.txt`](results_summary.txt).

## Repository layout

```
Nikhil_H_V/
├── Non_AI/
│   ├── count_non_ai.py      # classical pipeline
│   ├── requirements.txt
│   └── results/             # counts.csv + overlay images (downscaled copies)
├── AI/
│   ├── count_ai.py          # FastSAM pipeline
│   ├── requirements.txt
│   └── results/
├── ground_truth.csv         # true count per image (see "Ground truth")
├── evaluate.py              # compares counts.csv files with ground_truth.csv
└── results_summary.txt      # output of evaluate.py
```

## How to run

Download the dataset from the Google Drive link in the main README into `data/` at the repository root. The folder is git-ignored. You should end up with `data/ScrewAndBolt_20240713/` and `data/Screws_2024_07_15/`.

```bash
# Non-AI
cd Nikhil_H_V/Non_AI
pip install -r requirements.txt
python count_non_ai.py --data ../../data --out outputs

# AI (CPU is enough; FastSAM-s.pt, 23 MB, is downloaded automatically by ultralytics)
cd ../AI
pip install -r requirements.txt
python count_ai.py --data ../../data --out outputs

# Accuracy
cd ..
python evaluate.py Non_AI/outputs/counts.csv AI/outputs/counts.csv
```

Each script writes one overlay image per input to `outputs/`, with each instance in its own colour and the count in the corner, plus `outputs/counts.csv`.
Run times on a laptop CPU: Non-AI about 6 s per image, AI about 60 s per 12 MP image (tiled FastSAM). The AI script caches the raw FastSAM masks in `outputs/cache/`, so re-runs skip the model.

## 1. Non-AI solution (`Non_AI/count_non_ai.py`)

1. **Resize** so the long side is 2400 px.
2. **Flatten the illumination.** A large grey-level morphological closing removes the dark objects and leaves the background. The image is divided by that background, which removes vignetting (the `Screws_2024_07_15` set) and colour casts. Otsu's threshold then gives the foreground.
3. **Nut holes.** Enclosed bright regions that are round and of a consistent size are hex-nut holes. Each hole is one nut, which separates nuts that touch. The holes are trusted only when blob area divided by hole area is small (a ring). Bright reflections on bolt heads therefore don't count as holes.
4. **Discs.** Thick round nuts (bushings) often show no hole. Isolated round, solid blobs give the disc size. A Hough circle search at that radius then finds discs, including inside clusters.
5. **Group fragments.** Shiny screw heads break into pieces after thresholding. A small closing groups the pieces, but area is measured on the un-closed mask so that the gaps between touching objects don't inflate a cluster.
6. **Count per blob:**
   `holes + discs + round((area - holes*nut_area - discs*disc_area) / screw_area)`.
   The reference areas (nut, disc, screw) are estimated from each image itself with an iterated median that locks onto the most common blob size, i.e. one object. Nothing is hard-coded per image.
7. **Masks.** Each cluster is split into its k instances with marker-controlled watershed, seeded from the holes and from distance-transform peaks, with k-means as a fallback.

## 2. AI solution (`AI/count_ai.py`)

**Model:** [FastSAM-s](https://github.com/CASIA-IVA-Lab/FastSAM), a YOLOv8-seg network trained on Meta's SA-1B "Segment Anything" dataset (11 M images, 1.1 B masks), loaded through `ultralytics`.

**Training:** none. FastSAM is used zero-shot, because it segments every object without being told the class. The dataset has no labels, and 25 images is too few to train a detector honestly, so a pretrained class-agnostic segmenter is the more robust choice. If labels were available, the natural next step would be to fine-tune `yolov8n-seg` on tiles of these images, using the current masks as pseudo-labels after manual correction.

Pipeline:
1. **Tiling.** The image (long side 2400 px) is cut into 1024 px tiles that overlap by 256 px. The objects in the dense piles are only 30–40 px, so tiling keeps them large enough for the model.
2. **De-duplication.** Every mask is owned by the tile whose centre is nearest to the mask centroid. Each object therefore comes from exactly one tile, and no merging is needed, because the overlap is larger than any object.
3. **Filtering.** Masks that are not clearly darker than the background are dropped (background and shadow masks), as are tiny or huge masks.
4. **Mask hierarchy.** FastSAM returns whole objects, their parts (screw head, shank), near-duplicates and sometimes groups of touching objects. A containment graph is built, and starting from the top-level masks:
   - A mask much larger (more than 1.4×) than one object, with at least 2 object-sized sub-masks, is a group. It is replaced by its sub-masks, recursively.
   - Otherwise the mask is one object, and its sub-masks (parts) are discarded.
   - The size of "one object" is first taken as the median top-level mask, then re-estimated as the most common size among the selected masks, and the selection is repeated.
5. **Leftover merges.** Touching objects that FastSAM only returned as a single merged mask (at least 1.5× one object) count as `round(area / object_area)`. That mask is split with k-means for the overlay.
6. **Count** = number of instance masks.

## Ground truth

The dataset comes without labels, so `ground_truth.csv` was built as follows:

| Images | True count | How it was obtained |
|---|---|---|
| `Screws_2024_07_15/*` (7) | 43 | Counted by hand on the overlays. Every screw is separate, and the file `img1_43_nosy.jpg` confirms 43. |
| `192951`, `193135`, `193650`, `193659` | 21, 20, 10, 10 | Counted by hand on full-resolution crops. |
| `193831`, `193839`, `193907` (dense screws) | 191 | `193907` counted by hand on a grid of 20 full-resolution tiles. The other two are photos of the same pile reshuffled minutes apart. Both methods give near-identical counts across the three, which is consistent with that. |
| `194200` … `194316` (screws + round nuts) | 96–97 | `194200` (97) and `194316` (96) counted by hand on full-resolution tiles. The other four are the same set, so 96 was used for them. |
| `194541` … `194630` (hex nuts) | 355 | Independent nut-hole count (353–354 holes in every image, checked visually), plus one nut standing on its edge with no visible hole. |

Hand counts of dense piles carry roughly ±2 uncertainty, which is about 1 %.

## Known limitations

- `193135` (flanged bolts with mirror-like heads) is the worst image for both methods (22 against 20). Reflections split some heads in the thresholded mask, and FastSAM sees separate head and shank masks.
- In the mixed images, screws lying on top of round nuts are the main error source for the Non-AI method, because area-based splitting can't see occluded parts.
- The AI pipeline runs on CPU at about 1 minute per 12 MP image. A GPU, or FastSAM at a lower resolution for sparse images, would make it much faster.
