# Count the number of items in the image and overlay masks

## Overview
This project focuses on detecting and counting items in images, along with generating visual masks for detected items.

Two approaches have been implemented:

-  Non-AI Approach (Classical Computer Vision)
-  AI Approach (YOLOv8 Object Detection)

---

#  Approach 1: Non-AI (Classical CV)

##  Method
- Grayscale conversion  
- Thresholding (Adaptive / Otsu)  
- Morphological operations (Open + Close)  
- Connected Components Analysis  
- Area-based filtering  
- Watershed (for clustered objects)  

---

##  How to Run (Non-AI)

```bash
python Non_AI.py screws <input_image> [output_image]
```

```bash
python Non_AI.py nuts <input_image> [output_image]
```

### Example:

```bash
python Non_AI.py screws sample.jpg output.jpg
```

```bash
python Non_AI.py nuts sample.jpg output.jpg
```

---

## Calibration (Non-AI)

Calibration is used to estimate the area of a single object.

```bash
python Non_AI.py calibrate <input_image> [screws|nuts]
```

---

##  Why Calibration is Required

- Objects may overlap or touch each other  
- Multiple objects can appear as a single connected component  
- Calibration helps estimate how many objects exist in each blob  

---

# 🔹 Approach 2: AI (YOLOv8)

##  Model Details
- Model: YOLOv8s  
- Epochs: 150  
- Image Size: 1024x1024  
- Single-class detection used for improved counting  

---

##  Training

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    epochs=150,
    imgsz=1024
)
```

---

##  How to Run (AI)

```bash
python AI.py --model <path_to_model> --input <input_image> --output <output_image>
```

---

#  Installation

## Create virtual environment

```bash
python3 -m venv venv
```

## Activate it 
### Linux/Mac

```bash
source venv/bin/activate
```
### Windows

```bash
venv\Scripts\activate
```
### Install dependencies

```bash
pip install -r requirements.txt
```

---

#  Dataset

- Annotated dataset using Roboflow 
- Dataset not included as per instructions  


