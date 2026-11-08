# AI solution: YOLO11-seg trained on synthetic copy-paste data

An instance-segmentation network (Ultralytics **YOLO11s-seg**, COCO-pretrained) is fine-tuned to segment every screw, bolt or nut as a single class, `item`. The count is the number of predicted instances, and the predicted masks are drawn on the image.

## Run inference

The trained weights are **not committed** (see `.gitignore`). Train them as described below, or place a trained `best.pt` in `weights/`.

```bash
pip install -r requirements.txt          # install a CUDA build of torch first
python count_ai.py -i <ScrewAndBolt folder> <Screws folder> -o outputs --save-masks
python ../evaluate.py --pred outputs/counts.csv
```

Outputs:
- `outputs/overlays/*_overlay.jpg`: instance masks, outlines, ids and the count.
- `outputs/masks/*_labels.png`: 16-bit instance label maps.
- `outputs/counts.csv`: the count for each image.

The model was trained on 1024 px tiles resized to 512 px, i.e. at half the object scale. `count_ai.py` therefore resizes each photo to a 2000 px long side and runs the whole image at `imgsz = 2000 × 512 / 1024 ≈ 992` (rounded to a multiple of 32), the same object scale as in training. If you retrain at a different `imgsz`, pass `--train-imgsz`.

## Training method

### 1. Training data: copy-paste synthesis (`synth_data.py`)

The challenge provides no annotations, and hand-drawing polygons for about 4 000 touching or overlapping parts is impractical. The training data is therefore generated:

1. **Instance bank.** Items lying fully isolated in a training photo (one item per connected component, not touching the border) are cut out using the classical segmentation from `../Non_AI`. For isolated items that segmentation is pixel-exact. Each cut-out is stored as RGB + alpha, with nut holes transparent. This gave 611 cut-outs in total:

   | Set | Cut-outs |
   |---|---|
   | mixed screws and discs | 198 |
   | brass screws | 172 |
   | nuts | 123 |
   | small screws | 105 |
   | pan-head screws | 11 |
   | black screws | 2 |

2. **Backgrounds.** The same photos with every item removed by inpainting.
3. **Synthetic scenes.** Cut-outs are pasted onto random 1024 px background crops:
   - random rotation (0–360°), scale (±15 %), flip and brightness;
   - density from sparse to touching piles (nuts, placed by "growing" a pile) to overlapping heaps (screws, up to 50 % overlap);
   - soft edges, noise, blur and JPEG compression.

   Occlusion is tracked, so every visible mask is exact, even inside dense clusters. Items more than 70 % hidden are dropped.
4. **Real tiles.** 48 crops of training photos in which every item is isolated, so their labels are exact.

The result is 1 648 training tiles with about 254 000 labelled instances, and 200 synthetic validation tiles with about 30 000. These are re-used copies of the 611 real cut-outs: the variety lies in arrangement and occlusion, not in object appearance.

**Held-out photos.** One photo of each object set never contributes cut-outs, backgrounds or tiles:

`20240713_193135`, `20240713_193659`, `20240713_193907`, `20240713_194316`, `20240713_194630`, `img5`, `img6`, plus `img1_43_nosy` (a noisy copy of `img1`).

### 2. Fine-tuning (`train.py`)

| Setting | Value |
|---|---|
| Model | `yolo11s-seg.pt` (COCO-pretrained), 1 class |
| Epochs / batch / imgsz | 50 / 4 / 512 (tiles are 1024 px) |
| Optimiser | Ultralytics `auto`, cosine LR, AMP |
| Augmentation | mosaic, HSV, ±10° rotation, ±20 % scale, horizontal and vertical flips |
| Hardware / time | NVIDIA RTX 5060 Ti 16 GB, about 30 min |

Best epoch 34, on the synthetic validation set (masks): precision 0.986, recall 0.978, mAP50 0.993, mAP50-95 0.716. Training curves and settings are in `training_logs/` (`results.csv`, `args.yaml`).

To reproduce:

```bash
python synth_data.py --data <ScrewAndBolt folder> <Screws folder> --out dataset --n-train 1600 --n-val 200
python train.py --data dataset/data.yaml --epochs 50 --imgsz 512 --batch 4
```

`synth_data.py` uses a fixed seed, so it regenerates the same dataset. `train.py` copies the best checkpoint to `weights/best.pt`.

## Results on the real photos

Full table: [results.md](results.md).

| Subset | Images | Mean accuracy | Worst image | Exact counts |
|---|---|---|---|---|
| **Held-out photos** (never used for data) | 8 | **99.87 %** | 98.98 % | 7 / 8 |
| Photos used as cut-out sources | 17 | 99.82 % | 97.96 % | 15 / 17 |
| **All** | 25 | **99.84 %** | 97.96 % | 22 / 25 |

The remaining errors are +1 / +2 on the mixed screw-and-disc set, where screws lie on top of each other.

## Limitations

- **Little appearance variety.** Only 611 real cut-outs, and just 2 for the black screws; the model has seen few distinct object appearances.
- **Synthetic composites are not fully realistic.** They lack contact shadows, and the stacking isn't physically correct.
- **Training at 512 px halves the object scale.** Small screws are about 6 px wide at inference. Training on 512 px tiles (`synth_data.py --tile 512`) would keep full resolution at the same GPU memory cost.
- **Shared starting point.** The cut-outs come from the classical segmentation, so its errors on isolated items would carry over.
