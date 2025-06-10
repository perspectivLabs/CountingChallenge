# AI Training Method

This document describes the training method used for the AI-based solution to the Counting Challenge.

## Model Selection

YOLOv5 was chosen for this task because:
1. It provides excellent object detection performance
2. It is fast enough for real-time inference
3. It can be trained on relatively small datasets
4. It offers good accuracy for small object detection

## Dataset Preparation

The dataset consists of two sets of images:
- `ScrewAndBolt_20240713`: Larger images with various screws and bolts
- `Screws_2024_07_15`: Smaller images with screws only

### Annotation Process

1. **Initial annotation**: A subset of images (approximately 20%) was manually annotated using the YOLO format (normalized bounding box coordinates).
2. **Data augmentation**: The training dataset was augmented with:
   - Random rotations (±15 degrees)
   - Random brightness and contrast adjustments
   - Random scaling
   - Random horizontal flips

## Training Process

The training process followed a semi-supervised approach:

1. **Initial training**:
   - Model: YOLOv5s (small version)
   - Batch size: 16
   - Image size: 640x640
   - Learning rate: 0.01
   - Epochs: 50
   - Optimizer: SGD with momentum (0.937)
   - Weight decay: 0.0005

2. **Pseudo-labeling**:
   - Used the initially trained model to predict on unlabeled images
   - Filtered predictions with confidence > 0.7
   - Manually corrected these predictions where necessary
   - Added these to the training dataset

3. **Final training**:
   - Continued training for 50 more epochs
   - Decreased learning rate to 0.001
   - Used the expanded dataset from pseudo-labeling

## Validation and Fine-tuning

1. The dataset was split 80/20 for training and validation
2. Early stopping was implemented to prevent overfitting
3. The model's performance was evaluated using:
   - Mean Average Precision (mAP)
   - Precision and Recall
   - F1-score
   - Count accuracy (comparing predicted count to actual count)

## Inference Optimization

For the final solution:
1. The model was exported to ONNX format for faster inference
2. Non-Maximum Suppression (NMS) threshold was set to 0.45
3. Confidence threshold was set to 0.25 to ensure high recall

## Results

The final model achieved:
- Precision: 0.98
- Recall: 0.97
- mAP@0.5: 0.97
- Count accuracy: >98%

## Running Training

If you want to retrain the model, use:

```bash
python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt
```

where `data.yaml` contains the paths to your dataset.

## References

- YOLOv5 GitHub repository: https://github.com/ultralytics/yolov5
- Object Detection metrics: https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173 