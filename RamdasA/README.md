# CountingChallenge — RamdasA

Count objects (screws, bolts, hex nuts) in images and overlay segmentation masks.

## Results Summary

| Approach | Avg Accuracy | Method |
|----------|-------------|--------|
| Non-AI | 95%+ | OpenCV watershed + contour detection |
| AI | **96.3%** | YOLOv8n-seg fine-tuned |

## Folder Structure

```
RamdasA/
├── AI/
│   ├── ai_screw_counter.ipynb   # Inference notebook
│   ├── training_colab.ipynb     # Training notebook (run on Google Colab T4)
│   ├── best.pt                  # Trained YOLOv8n-seg weights (96.3% accuracy)
│   ├── requirements.txt
│   └── results/
│       ├── Screws_2024_07_15/
│       └── ScrewAndBolt_20240713/
└── Non_AI/
    ├── non_ai_screw_counter.ipynb
    ├── non_ai_screwbolt_counter.ipynb
    ├── requirements.txt
    └── results/
        ├── Screws_2024_07_15/
        └── ScrewAndBolt_20240713/
```

## Accuracy Results — AI (YOLOv8n-seg, conf=0.4, iou=0.3)

| Image | GT | Predicted | Accuracy |
|-------|----|-----------|----------|
| img1.jpg | 43 | 43 | 100.0% |
| img2.jpg | 43 | 43 | 100.0% |
| img3.jpg | 45 | 43 | 95.6% |
| img4.jpg | 45 | 43 | 95.6% |
| img5.jpg | 45 | 43 | 95.6% |
| img6.jpg | 45 | 43 | 95.6% |
| 20240713_192951.jpg | 21 | 21 | 100.0% |
| 20240713_193135.jpg | 22 | 21 | 95.5% |
| 20240713_193650.jpg | 10 | 10 | 100.0% |
| 20240713_193659.jpg | 10 | 10 | 100.0% |
| 20240713_193831.jpg | 194 | 195 | 99.5% |
| 20240713_193839.jpg | 188 | 191 | 98.4% |
| 20240713_193907.jpg | 186 | 188 | 98.9% |
| 20240713_194200.jpg | 85 | 96 | 87.1% |
| 20240713_194206.jpg | 85 | 95 | 88.2% |
| 20240713_194215.jpg | 95 | 95 | 100.0% |
| 20240713_194232.jpg | 90 | 95 | 94.4% |
| 20240713_194256.jpg | 88 | 90 | 97.7% |
| 20240713_194316.jpg | 88 | 90 | 97.7% |
| 20240713_194541.jpg | 425 | 360 | 84.7% |
| 20240713_194551.jpg | 431 | 355 | 82.4% |
| 20240713_194606.jpg | 374 | 361 | 96.5% |
| 20240713_194621.jpg | 367 | 356 | 97.0% |
| 20240713_194630.jpg | 365 | 355 | 97.3% |
| img_01.jpg | 354 | 361 | 98.0% |
| img_13.jpg | 354 | 356 | 99.4% |
| img_15.jpg | 354 | 361 | 98.0% |
| img_16.jpg | 354 | 355 | 99.7% |
| img_17.jpg | 354 | 357 | 99.2% |
| **Average** | | | **96.3%** |

## AI Approach

**Model:** YOLOv8n-seg (nano segmentation model)

**Training — Model 1 (best.pt, 96.3% accuracy):**
- 25 images manually annotated in LabelMe (polygon masks for `nut` and `screw` classes)
- Augmentation: flip_lr, flip_ud, rot90, brightness → **125 train + 25 val** images
- imgsz=1280, batch=2, epochs=10, mask_ratio=16, overlap_mask=False
- Best inference params: conf=0.4, iou=0.3

**Training — Model 2 (yolov8s, 300 images, 95% accuracy):**
- Same 25 source images with extended augmentation (flip_lr, flip_ud, flip_both, rot90, rot180, rot270, brightness up/down, contrast, gamma bright/dark)
- **300 train + 25 val** images at original resolution (~2212px)
- imgsz=2048, batch=1, epochs=50, YOLOv8s-seg on Google Colab T4
- More exact matches (11/29) but lower average due to density cap on 2 images

**Run inference:**
```bash
pip install -r AI/requirements.txt
jupyter notebook AI/ai_screw_counter.ipynb
```

## Non-AI Approach

**Pipeline:**
- Grayscale → Gaussian blur → Otsu threshold
- Morphological operations (open/close)
- Watershed segmentation for touching objects
- Hole detection for hex nuts (count inner holes)
- Contour filtering by area

**Run:**
```bash
pip install -r Non_AI/requirements.txt
jupyter notebook Non_AI/non_ai_screw_counter.ipynb
jupyter notebook Non_AI/non_ai_screwbolt_counter.ipynb
```
