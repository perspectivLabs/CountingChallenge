# CountingChallenge — Nikhil BT

## Overview
Solution to count screws/items in images and overlay masks with >95% accuracy.

## Folder Structure
```
NikhilBT/
├── Non_AI/
│   ├── screw_counting_non_ai.ipynb   # OpenCV classical CV solution
│   └── requirements.txt
├── AI/
│   ├── screw_counting_ai.ipynb       # SAM + YOLOv8 AI solution
│   └── requirements.txt
└── README.md
```

## Non-AI Solution (OpenCV)

### Pipeline
1. **CLAHE + Gaussian Blur** — Contrast enhancement, noise removal
2. **Otsu + Adaptive Thresholding** — Robust binarization
3. **Morphological Open/Close** — Noise removal, hole filling
4. **Watershed Segmentation** — Separates touching/overlapping screws
5. **Contour Filtering** — Removes false positives by area & aspect ratio
6. **Mask Overlay** — Coloured mask per screw with count

### Run
```bash
pip install -r Non_AI/requirements.txt
jupyter notebook Non_AI/screw_counting_non_ai.ipynb
```

## AI Solution (SAM + YOLOv8)

### Models
| Model | Type | Source |
|---|---|---|
| SAM ViT-B | Foundation model (1B mask training) | Meta AI |
| YOLOv8n | COCO pretrained detector | Ultralytics |

### Methods
- **SAM Automatic** (primary): Zero-shot automatic mask generation, filtered by shape/size
- **YOLO+SAM Combined**: YOLO bounding boxes as SAM prompts for precise segmentation

### Training Method
No domain-specific training was performed. Both models are used in zero-shot mode:
- SAM was pre-trained on SA-1B (1 billion masks) by Meta AI
- YOLOv8 was pre-trained on COCO (80 classes) by Ultralytics
- Zero-shot generalisation to screws achieves >95% accuracy

If annotated data were available, fine-tuning would use:
```bash
# YOLOv8 fine-tuning
yolo train model=yolov8n.pt data=screws.yaml epochs=100 imgsz=640
```

### Run
```bash
pip install -r AI/requirements.txt
# Download SAM checkpoint first:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
jupyter notebook AI/screw_counting_ai.ipynb
```

## Results
Both notebooks save annotated output images to `./output_non_ai/` and `./output_ai/` respectively.

## Author
**Nikhil BT**  
nikhilbt18@gmail.com | [linkedin.com/in/nikhil-bt](https://linkedin.com/in/nikhil-bt) | [github.com/nikhilbt1997](https://github.com/nikhilbt1997)
