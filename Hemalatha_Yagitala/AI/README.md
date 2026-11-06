# AI solution: YOLO11-seg with tiled inference

## Results (held-out test images, never used in training or tuning)
Accuracy per image = 1 − |predicted − actual| / actual. Ground truth: `../ground_truth.csv` (hand-counted).

**Mean count accuracy: 99.00% over 8 test images.**

| Test image | Predicted | Ground truth | Accuracy |
|---|---|---|---|
| 20240713_193135.jpg | 21 | 21 | 100.00% |
| 20240713_193659.jpg | 10 | 10 | 100.00% |
| 20240713_193907.jpg | 196 | 193 | 98.45% |
| 20240713_194232.jpg | 98 | 99 | 98.99% |
| 20240713_194316.jpg | 102 | 97 | 94.85% |
| 20240713_194621.jpg | 354 | 355 | 99.72% |
| img2.jpg | 43 | 43 | 100.00% |
| img5.jpg | 43 | 43 | 100.00% |

The test images were evaluated once with settings chosen on training images only. One image
falls just below 95%. It's reported as is instead of re-tuning on the test set.

## Run
```bash
pip install -r requirements.txt

# 1. Build the tiled dataset from images + instance masks (from ../Non_AI with --save-masks)
python make_dataset.py --images ../data --masks ../Non_AI/results/masks --out dataset

# 2. Train
yolo segment train model=yolo11n-seg.pt data=dataset/data.yaml imgsz=640 epochs=50 batch=8 patience=15 degrees=180 flipud=0.5 fliplr=0.5 device=cpu workers=0 project=runs name=seg exist_ok=True

# 3. Choose confidence and duplicate thresholds on TRAINING images only
python count_ai.py --weights weights/best.pt --input ../data --gt ../ground_truth.csv --sweep

# 4. Final evaluation on the held-out test images (+ overlays in results/overlays)
python count_ai.py --weights weights/best.pt --input ../data --output results --gt ../ground_truth.csv --test-only --conf 0.10 --dup 0.5
```
Trained weights: `weights/best.pt` (6 MB).

## Training method
| Step | Detail |
|---|---|
| Labels | Model-assisted labelling: instance masks produced by the Non-AI pipeline (`--save-masks`) were used as training labels. No hand-drawn masks |
| Preprocessing | Images resized to a 2000 px long side, cut into 640 px tiles with 160 px overlap. Objects mostly outside a tile are dropped from that tile's labels |
| Split | 8 held-out **test** images, one or more per photo group (`common.TEST_IMAGES`), excluded from training and tuning. The remaining 17 images are split by tile into train (150 tiles) and val (22 tiles) |
| Model | `yolo11n-seg`, COCO-pretrained, single class `object` |
| Hyperparameters | 640 px, batch 8, max 50 epochs with early stopping (patience 15), rotation ±180°, up/down and left/right flips, mosaic |
| Hardware | CPU only (Intel Core i5-4200U). Training stopped early at epoch 38, best at epoch 23 (~16.6 h) |
| Validation (best.pt) | Mask mAP50 0.968, mask mAP50-95 0.729. Box mAP50 0.974 |

## Inference
1. **Tiled prediction.** The image is cut into overlapping 640 px tiles. Each detection is kept only by
   the tile that *owns* its box centre (ownership boundaries at the middle of each overlap),
   so objects on tile seams are counted once.
2. **Mask-level duplicate removal.** Detections are sorted by confidence. A mask overlapping an
   already-kept mask by more than `--dup` of the smaller mask's area is dropped. Box NMS alone
   missed duplicates on elongated, diagonal screws.
3. **Fragment filter.** Masks smaller than 20% of the median object area are dropped.
4. **Threshold selection.** `--sweep` runs the model once per *training* image and scores a grid of
   confidence (0.05–0.30) × duplicate threshold (0.5–0.9). Best on training images: conf 0.10,
   dup 0.5 (98.99% mean on training images). These values were then applied once to the test images.

## Limitations and next steps
- The labels came from an earlier version of the Non-AI pipeline, so the model inherited some of its
  mistakes (for example, under-counting in tangled piles). Next steps: split by scene, hand-correct the
  training masks, then retrain.
- Training ran on a laptop CPU (Intel Core i5-4200U, ~16.6 h), so the small `yolo11n-seg` model was used.
  With a GPU, the next step would be the larger `yolo11s-seg` model and larger tiles.
- The best duplicate threshold (0.5) was the lowest value in the sweep grid. The next sweep should extend
  it downward (0.3–0.4). Six grid settings scored between 98.65% and 98.99% on training images, so the
  result doesn't depend on one lucky setting.
- The test images are new photos of the same piles seen in training, so 99.00% measures accuracy on new
  photos of familiar scenes, not on unseen scenes. A scene-level split is the right next evaluation.
- Validation mAP (0.968) measures agreement with the Non-AI labels, not with the true masks.
