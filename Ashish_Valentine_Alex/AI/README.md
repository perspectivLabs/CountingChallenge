# AI solution: YOLO26m-seg + SAHI

Notebook: [`counting_yolo26m_sahi.ipynb`](counting_yolo26m_sahi.ipynb) (Google Colab, T4 GPU): training in sections 1-7, SAHI inference and
accuracy in sections 8-12, with the training log and curves.
Requirements: [`requirements.txt`](requirements.txt). Training ran on Colab (Python 3.13); inference and the notebook's plumbing were
also run locally on Python 3.11.

## What it does
Counts the items in each image and overlays one polygon mask per counted item. Nuts, screws and bolts are one class
(`item`); the task asks for a count, not a per-type breakdown.

## Training method
- **Data:** the 25 provided photos (18 phone photos, about 2850-3000 x 4000, and 7 gold-screw photos, 1920x1080). Not committed.
- **Tiling:** each photo is cut into 1024x1024 tiles with 20% overlap (the last tile in a row/column is flush with the photo
  edge, so every tile is exactly 1024x1024). The same size and overlap are used for SAHI slices at inference.
- **Labels:** polygon labels in YOLO-seg format, one class (`item`). Only the visible part of an item is outlined, and an
  item cut by a tile edge is outlined only inside that tile. A first version of the labels came from the classical (Non_AI)
  pipeline and from hand annotation; every tile was then checked and corrected by hand in Label Studio (wrong masks deleted
  or reshaped, missing items added). Tiles with no items were left out (323 tiles).
- **Split (by whole photo, never by tile):** 17 photos for training and 8 for validation, 31% of the tiles. Validation
  photos: 193135, 193659, 193907, 194206, 194232, 194621, img2, img4. They cover every scene type (nuts, dense screw piles,
  medium-density screws, sparse screws, gold screws). Neighbouring tiles overlap, so a tile-level split would leak.
- **Offline augmentation (training photos only, before training):** every training tile has 6 augmented copies (flips,
  90-degree and small-angle rotations, light brightness/contrast change, noise and blur). Labels are transformed with the
  pixels. Validation tiles are original, unaugmented tiles only (1554 training tiles including the copies, 101 validation tiles).
- **Model:** `yolo26m-seg.pt` (COCO-pretrained), fine-tuned, single class, `imgsz=1024`, batch 4, up to 100 epochs, early
  stopping (patience 20). Training stopped at epoch 29 (best epoch 9, 2.1 h on a Colab T4, Ultralytics 8.4.204). Ultralytics' own
  augmentation (mosaic, flips, rotation, scale, HSV, ...) is switched off because the data is already augmented offline.
- **Tile-level validation (Ultralytics, 101 val tiles, mask):** precision 0.956, recall 0.915, mAP50 0.949, mAP50-95 0.735.
  These are per-tile detection scores, not count accuracy; the count accuracy on full photos is below. Curves:
  [`results/training_curves/`](results/training_curves/).

## Inference method (SAHI)
SAHI slices each full-resolution photo into 1024 px slices with 20% overlap and runs the model on every slice
(`image_size=1024`). Detections below a confidence threshold are dropped, and duplicates from overlapping slices are merged
by mask NMS: highest score first, and a mask is dropped if its overlap area is more than 20% of the smaller mask (IOS).
Measuring against the smaller mask also removes a small partial mask lying on a full one, which otherwise gives double
counts on large screws. The count is the number of masks left and the overlay is those masks.

**Confidence 0.35 and merge threshold 0.2 were chosen by a grid** (6 confidences x 5 merge thresholds, scored on the
validation photos; the notebook repeats a 4 x 4 version of it). This is a mild optimistic bias for the validation figure, because the same
photos are used for choosing the setting and for reporting. Nearby settings score 95-97% on the validation
photos (most of the grid is 93-97%), so the result is not a single lucky point. SAHI's built-in merge at 0.25/0.5 gave 92.5% on the validation photos.

## Accuracy definition
The task README does not define accuracy. Used here: per photo `1 - |predicted - true| / true` (clipped at 0), averaged
over photos, plus the exact-match rate. True counts: every tile was hand-checked, the tiles were stitched back onto the
full photo, and an item seen in two overlapping tiles was counted once (`ground_truth.csv`). Reported separately for
validation photos (held out from training) and training photos; **the headline figure is the validation photos.**

## Results (all 25 photos, `count_item_2.pt`, conf 0.35, merge 0.2)
| Split | Photos | Mean accuracy | Exact matches |
|---|---|---|---|
| Validation (held out) | 8 | **97.3%** | 3 |
| Training | 17 | 99.6% | 11 |
| All | 25 | 98.9% | 14 |

Validation photos: 193135 22/22, 193659 11/10, 193907 187/196, 194206 94/98, 194232 98/100, 194621 355/353, img2 43/43,
img4 43/43 (predicted/true). Per-photo results for all 25 photos are in `results/`.
Training-photo accuracy is optimistic because those photos were seen in training. The nuts (354) and gold screws (43)
are essentially exact. The lowest validation photos are 193659 (one extra mask on a 10-screw photo, 90%), 193907 (95.4%) and
194206 (95.9%).

## Honest limitations
- Several photos show the same objects rearranged (the 5 nut photos, the 7 gold-screw photos). A held-out photo has new
  arrangements and lighting but not new objects, so the validation score measures generalisation across arrangements.
- The validation set is small (8 photos) and also used to pick the confidence and merge thresholds.
- The ground truth was built from tile labels that started from the Non_AI masks and were corrected by hand, so errors
  that both methods share would not show up.
- Dense piles with heavy overlap are the hardest case.

## Standalone inference
`python infer_sahi.py --model count_item.pt --input <folder of original photos> --output <out folder>` runs the SAHI
step alone with the trained weights and the tuned defaults, and writes `overlays/<photo>.jpg`, `counts.csv` and
`polygons.json` (options: `--conf`, `--match`, `--device`, `--max-side`, `--limit`). Use a GPU; on a CPU it takes about a
minute per photo.

## How to run the notebook
1. In Google Drive put `augmented-correct-dataset-counting-split.zip` (the dataset) in `MyDrive/`, and the 25 original photos
   in `MyDrive/CountingChallenge/raw/` with `ground_truth.csv` (`image,count`) next to it.
2. Open the notebook in Colab with a GPU runtime and run all cells. Sections 1-7 train and save `count_item_2.pt`; sections
   8-12 run SAHI on the 25 photos and write overlays, `ai_counts.csv`, `ai_accuracy_per_image.csv` and polygons to
   `MyDrive/CountingChallenge/outputs/`.
3. To skip training, put the downloaded `count_item_2.pt` in `MyDrive/CountingChallenge/` and start at section 8.
