# AI solution: YOLO11-seg trained on Non_AI pseudo-labels

`count_objects_ai.py` runs a fine-tuned **YOLO11s-seg** instance-segmentation model on each image;
the count is the number of predicted instances, and every instance mask is drawn on the image.

```bash
pip install -r requirements.txt
python count_objects_ai.py --weights weights/best.pt --data ../../data --out output
python ../evaluate.py output/counts.csv
```

`weights/best.pt` (20 MB) is not committed. Recreate it with `make_dataset.py` (after running the
Non_AI script) and `train_colab.ipynb` (about 25 min on a free Colab T4). The overlays and counts
it produced are committed in `output/`.

## Training method

**Labels.** The dataset has no annotations, and hand-drawing ~4,000 polygons was not realistic in
the time available. Instead the masks from the Non_AI pipeline are used as training labels
(pseudo-labels, a form of knowledge distillation). They were improved first by auditing the
Non_AI output (see `../Non_AI/README.md`), because the network copies label mistakes: an early
model trained on rougher labels learned to output screw fragments.

**Data** (`make_dataset.py`)
* Every image resized to a 2000 px long side (the same working size as Non_AI), cut into
  640 x 640 tiles with 160 px overlap. An object enters a tile's labels only if >= 60 % of it is
  inside. Polygons are simplified with `approxPolyDP` (1 px).
* Split by **image**, never by tile:
  * test (never seen in training): `img5`, `193659`, `194206`, `194316`, `194621`, one or two of
    each scene type;
  * validation: `img3`, `193907`, `194551`;
  * train: the other 17 images -> 190 tiles, 4,240 instances (val: 35 tiles, 1,232 instances).

**Training** (`train_colab.ipynb`, Google Colab, Tesla T4)
* `yolo11s-seg.pt` (COCO pretrained), one class `object`, `imgsz=640`, batch 16, 100 epochs,
  early-stopping patience 30. Took 22 min; best epoch 94.
* Augmentation: rotation +-180 deg and both flips (parts lie at any angle), `scale=0.5` (screws in
  the sparse photos are ~2x larger than in the dense ones), mosaic, brightness jitter.
* Colab ships NumPy 2.x, which removed `np.trapz`; ultralytics 8.3.40 still calls it during
  validation, so the notebook maps it to `np.trapezoid`.
* Validation (mask): precision 0.946, recall 0.912, mAP50 0.959, mAP50-95 0.775.
  Curves and batches are in `training_run/`.

**Inference**
* The image is resized to the 2000 px working size and the model runs **once on the whole
  image** (`imgsz=2016`), so objects have exactly the scale seen in training.
* SAHI sliced inference was tried first and dropped: it cut masks at slice borders and found
  fewer objects in dense clusters than the single full-image pass.
* Confidence threshold **0.25**, chosen on the 12 non-test images with ground truth
  (0.20 -> 98.0 %, **0.25 -> 99.0 %**, 0.30 -> 98.3 %, 0.40 -> 96.8 % mean accuracy). The test
  images were not used to pick it.
* Memory: ultralytics upsamples all instance masks to full size at once (~4.5 GB of float32 for a
  354-nut image), which crashed an 8 GB laptop. The script keeps masks at the network's 1/4
  resolution and upscales each one only when drawing it; counts are unaffected.

## Results

Accuracy per image = `1 - |predicted - true| / true`, ground truth in `../ground_truth.csv`.

**Held-out test images** (never used for training or for choosing the threshold):

| Image | Type | Predicted | True | Accuracy |
|---|---|---|---|---|
| img5 | brass screws | 43 | 43 | 100 % |
| 20240713_193659 | sparse screws | 10 | 10 | 100 % |
| 20240713_194206 | mixed screws | 101 | 103 | 98.1 % |
| 20240713_194316 | mixed screws | 95 | 98 | 96.9 % |
| 20240713_194621 | nut pile | 354 | 354 | 100 % |
| **mean** | | | | **99.0 %** |

All 17 images with ground truth: mean 99.0 %, worst 93.1 % (`194200`, a training image with a
tight pile of overlapping screws: 95 vs 102).

## Limitations

* The model can only be as good as its labels; it learned from Non_AI masks, so where Non_AI is
  weakest (heavily overlapping screws) the model is too. It misses screws lying fully on top of
  each other (`194200`) and over-counts the densest screw-only photo slightly (205 vs 196).
* Next step with more time: hand-correct the pseudo-labels in the dense clusters (or label those
  tiles with SAM and review them) and retrain.
