# AI solution: YOLOv8s-seg trained on SAM 2.1 auto-labels

A YOLOv8s instance-segmentation model, fine-tuned on this dataset. It predicts one mask per object, and the count is the number of masks.

No human-made labels existed, so the labels were generated with **SAM 2.1** and checked by an automatic audit. A human corrected 16 objects in 8 of the 25 images, one click each.

```bash
pip install -r requirements.txt
# run from the repo root
python Om_Rode/AI/count_objects_yolo.py --data data --weights Om_Rode/AI/runs/yolov8s_seg/weights/best.pt
# -> Om_Rode/AI/outputs/<image>.jpg (masks + count) and outputs/counts.csv
python Om_Rode/evaluate.py Om_Rode/AI/outputs/counts.csv
```

The weights (`*.pt`) are not in git. Reproduce them with the pipeline below; training takes about 30 min on a 6 GB RTX 3050.

## Results

Accuracy per image is `1 - |pred - gt| / gt`, against `../ground_truth.csv`.

| split | images | mean accuracy | worst image |
|---|---|---|---|
| **test** (never seen in training) | 5 | **99.9 %** | 99.5 % |
| val | 2 | 99.5 % | 99.0 % |
| train | 18 | 99.3 % | 95.2 % |
| all | 25 | 99.4 % | 95.2 % |

The test split has one image of each scene type: sparse (img4), 10 screws, dense screws (193 objects), screws mixed with discs, and the nut pile (354 objects).

## Training method

### 1. Auto-labelling with SAM 2.1 (`autolabel_sam.py`)

- **Tiling.** Images are resized to a long side of 2400 px. SAM 2.1-base runs in "segment everything" mode on 1024 px tiles with stride 768. Masks that touch an interior tile edge are dropped, because the neighbouring tile sees those objects whole.
- **Clean-up.** The masks go through these steps:
  1. Remove duplicates by IoU.
  2. Keep only masks that lie on the dark foreground (from the Non-AI illumination-flattened threshold) and inside the valid field of view.
  3. Resolve group and part masks.
  4. Drop fragments smaller than 0.2 × the median object area.
- **Recovery of missed objects.** Up to 3 rounds. Each uncovered thick dark blob is prompted with a point, then a point plus a box. If the result is still too large, the blob gets one prompt per distance-transform peak, with the other peaks as negative points.
- **Merging parts.** SAM often splits a pan-head screw into head and shank. Adjacent part masks whose combined area is close to one object are merged.
- **Output.** YOLO polygon labels (class 0, "object"), compressed instance masks and overlays.

### 2. Audit and minimal human correction

- **Audit** (`audit_labels.py`). Each image is checked against independent classical signals:
  - **possible miss:** an uncovered thick foreground blob
  - **possible merge:** a mask larger than 1.6 × the median
  - **hole count:** one round hole equals one nut
  - **Non-AI count**
- **Correction** (`apply_corrections.py` + `corrections.json`). Only the flagged spots were looked at. 16 objects in 8 images were fixed with a single click each:
  - `add`: a SAM point prompt
  - `split`: one prompt per distance-transform peak, the others as negative points
  - `delete`
- **Sign-off.** The corrected labels were shown on a review page (`build_review_page.py`) and approved before training.
- **Outcome.** 17 of 25 images needed no correction. Before any correction, the SAM count was off by 0.9 % on average on the 18 nut, screw and disc images (worst 4.8 %).

### 3. Dataset (`prepare_dataset.py`)

- **Split by image** (no tile leakage):
  - test: `img4`, `193659`, `193907`, `194256`, `194621`
  - val: `img6`, `194206`
  - train: the remaining 18 images
- **Tiling.** Training images are cut into 1024 px tiles with stride 512 (182 tiles), at the same 2400 px scale used at inference. An object cut by a tile border is kept if at least 40 % of it is visible.

### 4. Training (`train.py`)

| setting | value |
|---|---|
| model | YOLOv8s-seg, COCO-pretrained (11.8 M params) |
| image size | 1024 |
| batch | 1 |
| epochs | 120 max, early stopping with patience 40 (stopped at 86, best epoch 46) |
| augmentation | rotation ±180°, vertical and horizontal flips, scale ±40 %, translation, HSV jitter |
| mosaic | off |
| other | `max_det=1000`, `overlap_mask=True` |

Mosaic is off and the batch is 1 for a reason. Nut tiles hold about 300 objects each, and mosaic would combine 4 of them, about 1,200 objects. That made the per-instance mask loss overflow the 6 GB GPU.

The validation mAP is lower than the counting accuracy (best mask mAP50 = 0.77). Validation runs on whole images downscaled to 1024 px, where the objects are about 2.3× smaller than at inference.

### 5. Inference (`count_objects_yolo.py`)

- Each image is resized to a long side of 2400 px and the whole image is predicted at once (`imgsz` rounded up to a multiple of 32).
- Settings: `conf=0.4`, `iou=0.5`, `max_det=1000`, retina masks.
- The count is the number of masks.

## Files

| file | purpose |
|---|---|
| `autolabel_sam.py` | SAM 2.1 tiled auto-labelling |
| `audit_labels.py` | flags possible misses and merges for human review |
| `apply_corrections.py`, `corrections.json` | one-click corrections |
| `build_review_page.py`, `review_notes.json` | label review page |
| `labels_sam/labels/*.txt` | final YOLO-seg labels (the annotations; the images are not committed) |
| `prepare_dataset.py`, `train.py` | dataset tiling and training |
| `count_objects_yolo.py` | inference: count + mask overlay |
| `../evaluate.py`, `../ground_truth.csv` | accuracy for both solutions |
