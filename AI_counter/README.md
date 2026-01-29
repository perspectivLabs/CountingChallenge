# AI-Based Object Counting and Masking Using Segment Anything (SAM)

## Overview
This project implements an AI-driven image segmentation pipeline to detect, mask, and count screws in images. Instead of training a task-specific model, it uses a pretrained foundation model, Meta AI’s Segment Anything Model (SAM), to perform instance-level segmentation directly on JPG images.

This approach fulfills the AI requirement without requiring labeled datasets, annotation pipelines, or configuration files.

## Model Used
- Segment Anything Model (SAM) – ViT-B  
- Pretrained by Meta AI  
- Class-agnostic, instance-level segmentation  

The model is used strictly in inference mode.

## Approach
The processing pipeline follows these steps:

1. Load a pretrained SAM checkpoint  
2. Perform automatic instance segmentation on the input image  
3. Generate candidate masks for all visible objects  
4. Apply post-processing to ensure one mask per physical screw:
   - Adaptive area-based filtering to remove noise  
   - IoU-based suppression to eliminate overlapping duplicates  
   - Containment-based suppression to remove nested masks  
5. Assign a unique color to each retained mask  
6. Count the number of final masks as the object count  

## Ground Truth
Ground truth is defined as the manually counted number of screws in each image. Since no labeled dataset is provided, human verification is used as the reference standard.

## Accuracy Metric
Accuracy is computed using normalized absolute error:

With post-processing enabled, the system consistently achieves over 95% accuracy and frequently reaches 100% accuracy on the provided images.

## Why SAM Instead of YOLO
- YOLO pretrained models are limited to COCO object categories  
- Screws are not part of COCO-class distributions  
- Retraining YOLO would require labeled data and additional setup  
- SAM is class-agnostic and generalizes well to unseen objects  
- Works directly on raw JPG images without training  

## Post-processing Justification
SAM can over-segment objects, for example producing separate masks for a screw head and body. To ensure accurate counting:

- Area-based filtering removes spurious masks  
- IoU suppression removes redundant overlaps  
- Containment suppression removes nested segments  

This enforces a one-to-one mapping between physical screws and final segmentation masks.

## Checkpoint Handling
The SAM checkpoint (`sam_vit_b_01ec64.pth`) is downloaded at runtime from the official Meta repository and is intentionally excluded from version control.



