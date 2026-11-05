# AI solution: fine-tuned YOLO11n-seg

Counts the screws, bolts and nuts in each image and draws one mask per object, using a YOLO11n-seg
instance-segmentation model fine-tuned on this dataset. There is one class, `item`.

## Run

Python 3.12, managed with [uv](https://docs.astral.sh/uv/). `requirements.txt` is exported from `uv.lock` for pip users.
Runs on Apple MPS, CUDA or CPU. The trained model is `model/items_yolo11n_seg.pt` (6 MB).

```bash
cd Saravana_Perumal_Natarajan/AI
uv sync
uv run count.py /path/to/data -o outputs      # overlays + counts.json
uv run evaluate.py /path/to/data -o outputs   # scores against ../ground_truth.json, test split shown separately
```

`ai_counting.ipynb` shows the training data, the training curves and the held-out test images.

Retrain from scratch:

```bash
(cd ../GroundTruth && uv sync && uv run sam_count.py /path/to/data)   # SAM 2 masks used for labelling
uv run make_dataset.py /path/to/data                                  # tiles + labels -> datasets/items
uv run train.py                                                       # -> runs/segment/items/weights/best.pt
```

## Training method

1. **Labels.** The dataset comes without labels, so none were drawn by hand. They come from the
   ground-truth step (`../GroundTruth`):
   - Objects that the Non-AI counter and SAM 2 agree on (one-to-one mask IoU ≥ 0.3) get the SAM 2 mask.
     When SAM 2 outlined only part of an object (a screw head without its thread), the union of both masks is used.
   - Disputed objects accepted on review get the instance under their stored point.
2. **Split by image.** Each group has one image in validation and one in test:

   | Split | Images |
   |---|---|
   | test (held out) | `img2`, `193659`, `193907`, `194316`, `194630` |
   | val (model selection) | `img5`, `193650`, `193839`, `194256`, `194621` |
   | train | the other 15 images |
3. **Tiles.** Images are resized so the long side is 2000 px and cut into 640 px tiles with 160 px overlap.
   - An object becomes a polygon in every tile that holds at least 15% of it.
   - Most empty tiles are dropped.
   - Result: 152 train and 50 val tiles (`make_dataset.py`).
4. **Fine-tuning** (`train.py`): COCO-pretrained `yolo11n-seg.pt` at 640 px.
   - Batch 8, up to 120 epochs, early stopping with patience 40.
   - Augmentation: mosaic, rotation up to 180°, horizontal and vertical flips.
   - Training stopped at epoch 89, best at 49, after 26 minutes on an Apple M1 Pro.
5. **Inference** (`count.py`). The same tiling is used.
   - A detection counts only if its mask centre lies in the tile's core, so objects in the overlap are counted once.
   - Masks are painted highest-confidence first. A mask that is already mostly covered is a duplicate and is dropped.
   - The confidence threshold of 0.35 was chosen on the validation images.

## Results

Scored against `../ground_truth.json`. Count accuracy = 1 − |predicted − true| / true.

| Split | Images | True / predicted | Mean accuracy | Worst image | Total-count accuracy |
|---|---|---|---|---|---|
| **test (held out)** | 5 | 699 / 698 | **99.90%** | 99.48% | 99.86% |
| val | 5 | 701 / 698 | 99.69% | 98.47% | 99.57% |
| train | 15 | 1907 / 1903 | 99.35% | 95.24% | 99.79% |
| **all** | **25** | **3307 / 3299** | **99.53%** | 95.24% | 99.76% |

| Test image | True | Predicted |
|---|---|---|
| `Screws_2024_07_15/img2.jpg` (brass screws) | 43 | 43 |
| `20240713_193659.jpg` (sparse screws) | 10 | 10 |
| `20240713_193907.jpg` (dense screw pile) | 194 | 193 |
| `20240713_194316.jpg` (screws + bolt heads) | 98 | 98 |
| `20240713_194630.jpg` (hex nuts) | 354 | 354 |

The weakest image is `193135`, a training image: 22 counted for 21 large pan-head screws. Only two images
contain that screw type. [`results/`](results/) holds test overlays, training curves and per-image scores.

The submitted model is the third training run. Two changes made the difference:
- In the first run's labels, some screws were outlined by their head only. Using the union of both masks fixed that.
- In the second run, disputed objects were looked up by a stale instance id. They are now looked up by their stored point.

The run was chosen on validation accuracy:

| Run | Validation | Test |
|---|---|---|
| 1 | 98.03% | 98.65% |
| 2 | 99.39% | 99.18% |
| 3 (submitted) | 99.69% | 99.90% |

## Limitations

- **Small, correlated data.** Each group shows the same objects photographed several times. Test images are new
  arrangements of known objects, not new object types.
- **Labels and ground truth share one source:** Non-AI and SAM 2 agreement plus review. Scores measure
  agreement with that reference (see `../GroundTruth/README.md`).
- **Tangled screw piles are hardest.** Errors are screws split into head and thread, or two crossing screws
  merged into one.
