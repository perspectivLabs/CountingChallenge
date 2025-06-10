# Counting Challenge

This repository contains solutions to the Counting Challenge, which involves counting the number of items (screws/bolts) in images and overlaying masks for the same.

## Task Definition

- Count the number of items in the image and overlay masks for the same
- Achieve accuracy > 95%

## Solutions Provided

This repository contains two solutions to the challenge:

1. **Non-AI Solution** (in the `Non_AI` folder):
   - Uses classical computer vision techniques with OpenCV
   - Implements both Otsu's and Adaptive thresholding methods
   - Selects the best result for each image

2. **AI Solution** (in the `AI` folder):
   - Uses YOLOv5, a deep learning object detection model
   - Bootstraps training with a semi-supervised approach
   - Provides more robust results on complex images

## Dataset

The dataset consists of two folders of images:
- `ScrewAndBolt_20240713` - Contains larger images with various screws and bolts
- `Screws_2024_07_15` - Contains smaller images with screws

## Requirements

Each solution has its own requirements file. Please refer to the respective README files in each solution folder for detailed instructions.

## How to Run

Please refer to the README files in each solution folder for detailed instructions on how to run the respective solutions.

## Results

Both solutions generate:
- Annotated images with object masks
- Count information displayed on the images
- Console output summarizing the results

The AI solution generally performs better on:
- Images with poor lighting conditions
- Images with overlapping objects
- Complex backgrounds

## Comparison between Non-AI and AI Solutions

| Aspect | Non-AI Solution | AI Solution |
|--------|----------------|-------------|
| Setup time | Fast (no training required) | Slower (requires training) |
| Processing speed | Very fast | Moderate (faster during inference) |
| Accuracy | Good on clean images | Better overall, especially on complex images |
| Robustness | Sensitive to lighting, overlap | More robust to variations |
| Scalability | Limited to similar images | Can generalize to new images |

## Submission Guidelines

To submit your solution:

1. Fork this repository
2. Create a root folder with your name as per your resume
3. Inside your named folder, create two subfolders:
   - `AI/` - For the AI-based solution
   - `Non_AI/` - For the Non-AI solution
4. Add all necessary code, requirements files, and documentation
5. Do NOT include dataset files in your submission
6. Clean up any unwanted files by adding them to `.gitignore`
7. Raise a PR with your name to this repository from your forked repository

## Cleanup Guidelines

Before submitting your PR, ensure you have:
- Removed all dataset files
- Removed result folders (they can be regenerated)
- Removed any model weight files (*.pt, *.weights, *.onnx)
- Removed any unnecessary files (e.g., __pycache__, .DS_Store)
- Added proper documentation for your solutions 