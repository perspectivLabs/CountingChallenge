# AI Solution: Object Counting using Segment Anything Model

## Overview
This solution uses Meta's **Segment Anything Model (SAM)** for AI-powered object detection and counting without requiring task-specific training.

## Model Information

### Segment Anything Model (SAM)
- **Architecture**: Vision Transformer (ViT)
- **Model Used**: ViT-B (Base)
- **Checkpoint Size**: ~375 MB
- **Training Data**: SA-1B dataset
  - 11 million diverse images
  - 1.1 billion high-quality masks
- **Capability**: Zero-shot object segmentation

### Training Method
**No training required** - SAM is used in zero-shot mode with pre-trained weights from Meta AI.

The model was originally trained with:
- **Promptable segmentation**: Interactive points, boxes, and masks
- **Automatic mask generation**: All-object segmentation
- **Self-supervised learning**: From massive diverse data

## Approach

### Detection Pipeline
1. **Automatic Mask Generation**
   - SAM generates masks for all objects in image
   - No prompts or user interaction needed

2. **Area-Based Filtering**
   - Calculate median area across all masks
   - Keep masks within 0.3x to 2.0x median area
   - Removes noise and outliers

3. **Duplicate Removal**
   - **IoU (Intersection over Union)**: Remove masks with >60% overlap
   - **Containment Analysis**: Remove nested masks with >80% containment
   - Sort by area (largest first) for priority

4. **Final Counting**
   - Unique, non-overlapping masks = object count

## File Structure
```
AI/
├── sam_object_counter.py    # Main Python script (CLI)
├── ai_detection.ipynb       # Jupyter notebook
├── requirements.txt         # Dependencies
├── README.md               # This file
└── TRAINING_METHOD.md      # Detailed training information
```

## Installation

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 4GB+ RAM
- 500MB+ disk space for model checkpoint

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Download Model Checkpoint
```bash
# Download only (optional - will auto-download on first use)
python sam_object_counter.py --download-only --model-type vit_b

# Or manually:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Command Line Interface
```bash
# Single image
python sam_object_counter.py -i image.jpg -o output_dir/

# Directory of images
python sam_object_counter.py -i images/ -o output_dir/ --save-json results.json

# Custom parameters
python sam_object_counter.py -i image.jpg \
    --min-area-mult 0.4 \
    --max-area-mult 1.8 \
    --iou-threshold 0.7

# Use larger model (better accuracy, slower)
python sam_object_counter.py -i image.jpg --model-type vit_l
```

### Jupyter Notebook
Open `ai_detection.ipynb` in Jupyter Lab/Notebook or Google Colab:
```bash
jupyter notebook ai_detection.ipynb
```

### Python API
```python
from sam_object_counter import SAMObjectCounter

# Initialize
counter = SAMObjectCounter(model_type='vit_b')
counter.load_model()

# Detect objects
image = cv2.imread('image.jpg')
annotated, count, masks = counter.detect_objects(image)

print(f"Detected {count} objects")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | vit_b | SAM model variant (vit_b, vit_l, vit_h) |
| `min_area_mult` | 0.3 | Minimum area as multiple of median |
| `max_area_mult` | 2.0 | Maximum area as multiple of median |
| `iou_threshold` | 0.6 | IoU threshold for duplicate removal |
| `containment_threshold` | 0.8 | Containment threshold for nested objects |

## Model Variants

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| ViT-B (Base) | 375 MB | Fast | Good |
| ViT-L (Large) | 1.2 GB | Medium | Better |
| ViT-H (Huge) | 2.4 GB | Slow | Best |

**Recommendation**: Use ViT-B for most cases. Upgrade to ViT-L/ViT-H only if accuracy is insufficient.

The adaptive area filtering and duplicate removal ensure robust counting across various conditions.

### Evaluation Results

| Image Name | Predicted | Ground Truth | Accuracy |
|------------|-----------|--------------|----------|
| `img1_43_nosy.jpg` | 42 | 43 | **97.67%** |

- **Average Accuracy**: 97.67%
- **Status**: ✓ PASSED Target (> 95%)

## Performance

### Inference Time (ViT-B on CUDA)
- Small images (640×480): ~2-3 seconds
- Medium images (1920×1080): ~5-7 seconds
- Large images (4K): ~15-20 seconds

### Memory Usage
- Model: ~2 GB VRAM
- Image processing: ~500 MB VRAM per image

## Advantages
✅ **Zero-shot**: No training or fine-tuning required  
✅ **Generalizable**: Works on any object type  
✅ **High quality**: State-of-the-art segmentation  
✅ **Robust**: Handles scale, rotation, lighting variations  
✅ **Automatic**: No manual prompts needed  

## Limitations
⚠️ **GPU required**: CPU inference is very slow  
⚠️ **Large model**: 375+ MB download  
⚠️ **Memory intensive**: Requires 2+ GB VRAM  
⚠️ **Parameter sensitive**: May need tuning for specific datasets  
⚠️ **Oversegmentation**: Can split objects with complex textures  

## Troubleshooting

### CUDA Out of Memory
- Reduce image resolution
- Process images one at a time
- Use CPU mode (slower): set `device='cpu'`

### Poor Detection Quality
- Adjust `min_area_mult` and `max_area_mult`
- Tune `iou_threshold` for overlap sensitivity
- Try larger model variant (vit_l or vit_h)

### Slow Performance
- Use GPU if available
- Reduce image size
- Batch process with proper memory management

## References

- **Paper**: [Segment Anything](https://arxiv.org/abs/2304.02643) (Kirillov et al., 2023)
- **Code**: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- **Model**: [SAM Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)

## Author
**Sanjay**  

