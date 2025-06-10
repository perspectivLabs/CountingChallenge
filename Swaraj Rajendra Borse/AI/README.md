# AI Object Counting Solution

This solution uses AI techniques to count objects in images and overlay masks on them. Two implementations are provided:

## 1. Pre-trained YOLOv5 Approach (Recommended)

This approach uses a pre-trained YOLOv5 model with a fallback to classical computer vision when needed:

### How it Works

1. **Pre-trained Model Detection**:
   - Uses YOLOv5s trained on COCO dataset
   - Detects objects that could be similar to screws/bolts
   - Filters by confidence threshold and relevant object classes

2. **Hybrid Verification**:
   - Verifies AI-based results with classical computer vision methods
   - Falls back to Non-AI count if AI detection fails or is significantly different
   - Ensures reliable counting even when AI is uncertain

### How to Run

```
python count_objects_pretrained.py
```

## 2. Custom-Trained YOLOv5 Approach (Optional)

This approach trains a custom YOLOv5 model on your specific dataset:

### How it Works

1. **Dataset Preparation**:
   - Automatically creates a dataset for YOLOv5 training
   - Generates initial labels using classical computer vision techniques

2. **Training**:
   - Trains YOLOv5 on the prepared dataset
   - Saves the best weights for inference

3. **Inference**:
   - Uses the trained model to detect objects in images
   - Creates mask overlays and counts objects

### How to Run

```
python count_objects_yolo.py
```

**Note**: This approach requires more setup and training time, and may encounter issues with path handling in Windows. The pre-trained approach is generally more reliable.

## Requirements

```
torch>=1.7.0
torchvision>=0.8.1
numpy>=1.18.5
opencv-python>=4.1.2
Pillow>=7.1.2
PyYAML>=5.3.1
tqdm>=4.41.0
```

## Output

Both solutions generate:
- Annotated images showing detected objects with colored masks
- Count information directly on the images
- Results saved in separate directories for each dataset

## Installation

```
pip install -r requirements.txt
``` 