# Counting Challenge — Ramesh K

Count the items (screws, bolts, nuts, washers) in each image and overlay a mask on every counted item, using
1. a **Non-AI** method (classical computer vision, OpenCV), and
2. an **AI** method (fine-tuned YOLO11 instance segmentation).

```
Ramesh_K/
├── ground_truth.csv              reference counts used for accuracy
├── Non_AI/
│   ├── non_ai_counting.ipynb     classical CV solution (executed, outputs included)
│   └── requirements.txt
└── AI/
    ├── ai_counting.ipynb         pseudo-labels → training → tiled inference (executed, outputs included)
    └── requirements.txt
```

## Results

Accuracy per image = `1 − |predicted − true| / true`.

| Image | True | Non-AI | AI | AI split |
|---|---:|---:|---:|---|
| ScrewAndBolt/20240713_192951 | 21 | 21 | 21 | train |
| ScrewAndBolt/20240713_193135 | 21 | 21 | 21 | **test** |
| ScrewAndBolt/20240713_193650 | 10 | 10 | 10 | train |
| ScrewAndBolt/20240713_193659 | 10 | 10 | 10 | **test** |
| ScrewAndBolt/20240713_193907 | 195 | 195 | 184 | **test** |
| ScrewAndBolt/20240713_194316 | 98 | 98 | 85 | **test** |
| ScrewAndBolt/20240713_194630 | 354 | 354 | 354 | **test** |
| Screws/img1, img1_43_nosy, img2, img3 | 43 each | 43 | 43 | train |
| Screws/img4, img5 | 43 each | 43 | 45 | train |
| Screws/img6 | 43 | 43 | 43 | **test** |

| Solution | Images | Mean accuracy | Total-count accuracy | Exact |
|---|---|---:|---:|---:|
| Non-AI | 14 labelled images | 100.00 % | 100.00 % | 14 / 14 |
| AI | 6 held-out test images | **96.85 %** | **96.67 %** | 4 / 6 |
| AI | 14 labelled images | 97.98 % | 97.23 % | 10 / 14 |

Counts for the 9 images without a reference count (e.g. `194541`: Non-AI 354, AI 354; `193831`: Non-AI 195, AI 182) are printed in both notebooks, and both notebooks save an overlay image for every photo.

### Reference counts (`ground_truth.csv`)
The dataset ships without labels.
* `Screws_2024_07_15/*` — 43 each, taken from the file name `img1_43_nosy.jpg` (the seven photos show the same 43 screws).
* 7 `ScrewAndBolt` images, including the 5 AI test images — counted by hand on numbered overlays: isolated items were marked, every touching cluster was labelled with its estimated count, and each label was checked by eye.

Because the overlays used for manual checking came from the Non-AI pipeline, its 100 % on those 7 images is not a fully independent measurement. The AI numbers are independent of this, because the test images were never used for its labels or training.

## Non-AI solution (`Non_AI/non_ai_counting.ipynb`)
OpenCV, NumPy and SciPy only. No learned parameters.

1. **Resize** each photo so its long side is at most 2400 px.
2. **Background flattening.** A 3rd-order polynomial surface is fitted to background pixels and divided out. This removes uneven light, coloured paper and lens vignetting.
3. **Otsu threshold and morphological opening.** Holes are filled so a nut becomes one solid blob.
4. **Unit area.** The area of a single item is estimated per image with an iterated median of blob areas, so there are no hard-coded sizes.
5. **Mode detection.** If more than 70 % of single-sized blobs contain one round hole, the image is a nut/washer image.
6. **Splitting touching items:**
   * *Nuts/washers:* every round, correctly sized hole is one item. Each item's mask is the part of the blob nearest to its hole (Voronoi split via a distance transform, limited to one nut radius). This counts piles of 350+ touching nuts exactly.
   * *Screws/bolts:* items in a blob = `round(blob_area / unit_area)`. The blob is split into that many masks with k-means seeds and a watershed on the distance transform.
7. Blobs touching the image border are vignetting, not items, and are ignored.

Runtime: about 2.5 min for all 25 images on a CPU.

## AI solution (`AI/ai_counting.ipynb`)

### Training method
| | |
|---|---|
| Base model | `yolo11s-seg.pt`: Ultralytics YOLO11-small instance segmentation, COCO-pretrained (generic objects, no screw/nut data) |
| Labels | **Automatic pseudo-labels** from the classical pipeline, run on the training images only. Only *confident* instances are kept: isolated single-sized items, and nuts/washers split by their own hole. Uncertain regions (touching screws) are painted with the background colour so they are neither positives nor missed negatives. No manual polygon drawing. |
| Split | 6 held-out test images, one per object group (`193135, 193659, 193907, 194316, 194630, img6`), never used for labels or training. The other 19 images were cut into tiles, with 85 / 15 % train/val by tile. |
| Data | Images scaled to a 2400 px long side and cut into 640×640 tiles with a 480 px stride: 304 tiles, 2,075 labelled items (3,604 train + 762 val instances including partial items at tile edges). Single class `item`. |
| Hyper-parameters | 40 epochs, imgsz 640, batch 2 (Ultralytics accumulates to a nominal batch of 64), optimizer auto (AdamW, lr 0.002), seed 0, deterministic, patience 25 |
| Augmentation | Horizontal and vertical flips, ±90° rotation, scale/translate, HSV jitter. Mosaic and copy-paste were off, because 4 dense nut tiles per sample exhausted the 4 GB GPU. |
| Hardware / time | NVIDIA GTX 1650 4 GB, 0.81 h of training. Also runs unchanged on a Kaggle/Colab T4 (batch 16 default). |
| Validation | box mAP50 0.995, mask mAP50 0.995 on validation tiles |

### Inference
1. Scale the image to 2400 px and predict 640×640 tiles with a 160 px overlap (confidence 0.25).
2. Drop detections cut by an inner tile edge. The overlap is larger than any item, so the item is complete in the neighbouring tile.
3. Remove the remaining duplicates when masks overlap by more than 50 % of the smaller mask.
4. Count = number of kept instances. The masks are painted into one label map for the overlay.

### Known limitation
Touching or overlapping screws are excluded from the pseudo-labels, so the model has seen few examples of them. It undercounts dense screw piles (`193907`: 184 of 195; `194316`: 85 of 98). Hand-labelling some touching-screw clusters, or self-training on the model's own confident predictions, would be the next step.

## How to run
Python 3.8.3 was used (3.8+ works).

```bash
pip install -r Non_AI/requirements.txt    # or AI/requirements.txt
```
1. Download the dataset and unzip it so the images are at `data/CountingChallenge/ScrewAndBolt_20240713/` and `data/CountingChallenge/Screws_2024_07_15/` (the `data/` folder is at the repo root and git-ignored). Alternatively, set the environment variable `DATA_DIR`. On Kaggle the dataset is found automatically under `/kaggle/input`.
2. Open the notebook from its own folder and **Run All**.
   * Non-AI: outputs go to `Non_AI/outputs/` (overlay per image and `counts.csv`).
   * AI: builds the dataset, trains, then counts. Outputs go to `AI/work/` (weights at `work/best.pt`, overlays and `counts.csv` in `work/outputs/`). Environment variables `EPOCHS` and `BATCH` override the defaults (batch 2 fits a 4 GB GPU). Without a GPU it runs a 1-epoch smoke test.

Generated files (`outputs/`, `AI/work/`, `*.pt`) and the dataset are excluded via `.gitignore`.
