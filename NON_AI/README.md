# Non-AI Object Counting and Masking (OpenCV)

## Overview
This solution performs object counting and mask generation using **traditional computer vision techniques** without any machine learning or AI models. The goal is to detect, segment, and count screws present in an image using classical image processing methods.

## Approach
The Non-AI pipeline is implemented using OpenCV and follows these steps:

1. Convert the input JPG image to grayscale
2. Apply contrast enhancement and Gaussian blurring
3. Perform adaptive thresholding to separate screws from background
4. Apply morphological operations to remove noise
5. Detect contours corresponding to individual screws
6. Filter contours based on area to remove false positives
7. Generate binary masks and overlay contours on the original image
8. Count the number of valid contours as the object count

## Ground Truth
For the Non-AI approach, **ground truth is obtained by manually counting** the number of screws visible in each JPG image. This manually verified count serves as the reference for accuracy calculation.

## Accuracy Metric
Accuracy is computed using normalized absolute error:

Accuracy = 1 − |Predicted Count − Ground Truth| / Ground Truth

The approach achieves **>95% accuracy** for images with controlled lighting and minimal object overlap.

