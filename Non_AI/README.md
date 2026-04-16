# Non-AI Solution: Screw and Bolt Detection

## Overview
This solution uses traditional Computer Vision techniques (OpenCV) to detect and count screws and bolts in images without using any AI/ML models.

## Approach

### Key Techniques
1. **Preprocessing**
   - Grayscale conversion
   - Gaussian blur for noise reduction
   - Binary thresholding (THRESH_BINARY_INV)

2. **Morphological Operations**
   - Closing: Fill small gaps in objects
   - Opening: Remove small noise particles

3. **Contour Detection**
   - External contours for object boundaries
   - Area-based filtering (100-10000 pixels)

4. **Classification Logic**
   - **Bolts**: Detected by center hole presence
   - **Screws**: Objects without center holes
   
5. **Shape Analysis**
   - Hexagonal detection via polygon approximation
   - Circular detection via circularity metric: `4πA/P²`

## File Structure
```
Non_AI/
├── screw_bolt_detector.py    # Main Python script (CLI)
├── non_ai_detection.ipynb    # Jupyter notebook
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line
```bash
# Single image
python screw_bolt_detector.py -i path/to/image.jpg -o output_dir/

# Directory of images
python screw_bolt_detector.py -i path/to/images/ -o output_dir/ --save-json results.json

# Custom parameters
python screw_bolt_detector.py -i image.jpg --min-area 150 --max-area 8000 --threshold 70
```

### Jupyter Notebook
Open `non_ai_detection.ipynb` in Jupyter/Colab and run all cells.

### Python API
```python
from screw_bolt_detector import ScrewBoltDetector

detector = ScrewBoltDetector(min_area=100, max_area=10000)
image = cv2.imread('image.jpg')
annotated, screws, bolts, detections = detector.detect_objects(image)
print(f"Found {screws} screws and {bolts} bolts")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_area` | 100 | Minimum contour area (pixels) |
| `max_area` | 10000 | Maximum contour area (pixels) |
| `threshold` | 60 | Binary threshold value |
| `blur_kernel` | (5, 5) | Gaussian blur kernel size |

The solution achieves high accuracy on well-lit images with clear object separation. 

### Evaluation Results
- **Total Objects Detected**: 215
- **Screws**: 84
- **Bolts**: 131
- **Accuracy**: **95%** (Target > 95%)

Performance varies based on:
- Lighting conditions
- Object overlap
- Background contrast
- Image quality

## Advantages
✅ Fast inference (no model loading)  
✅ Lightweight (no GPU required)  
✅ Interpretable logic  
✅ Easy parameter tuning  

## Limitations
⚠️ Sensitive to lighting  
⚠️ Requires parameter tuning per dataset  
⚠️ Struggles with heavy overlap  

## Author
**Sanjay**  

