# Counting Challenge

## Overview

This project addresses the challenge of counting the number of screws and bolts in images and overlaying masks to visualize the detected items.

The solution is implemented using both non-AI and AI techniques. 

## Project Structure

- `non_ai/` - Contains the solution using non-AI techniques.
- `ai/` - Contains the solution using AI techniques.

## Non-AI Solution

The non-AI solution utilizes OpenCV for image processing. It involves:

1. **Preprocessing**:
   - Grayscale conversion
   - Gaussian blur

2. **Thresholding**:
   - Adaptive or Otsu's thresholding to obtain binary images

3. **Morphological Operations**:
   - Dilation and erosion to enhance object shapes

4. **Contour Detection**:
   - Contour detection to count screws and bolts

5. **Overlaying Masks**:
   - Drawing contours on the original image to visualize detected items


## AI Solution Overview

The AI solution leverages two YOLO models for object detection and segmentation:

## YOLO Object Detection

1. **Model Used**: YOLOv8s-World
   - **Description**: This model is fine-tuned for general object detection tasks. We use this model to detect screws in images.
   - **Features**:
     - Custom class: Screw
     - Prediction with high confidence threshold
   - **Process**:
     - Load the YOLOv8s-World model.
     - Set custom classes and run predictions on images.
     - Count detected screws and visualize the results with overlay masks.

## YOLO Segmentation

1. **Model Used**: YOLOv8n-Seg
   - **Description**: This model is used for semantic segmentation to detect and segment screws and bolts in images.
   - **Features**:
     - Fine-tuned on a dataset of nuts and bolts
     - Capable of providing detailed segmentation maps
   - **Process**:
     - Train the YOLOv8n-Seg model using a dataset from Roboflow.
     - Predict and annotate the images with class counts.
     - Display results with annotations and segmentation masks.

