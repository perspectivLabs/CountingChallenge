# Counting Challenge Solution

This is my solution for the Counting Challenge, which involves counting the number of items in images and overlaying masks for the same with >95% accuracy.

## Project Structure

- **AI/** - Contains the AI-based solution using YOLOv5
- **Non_AI/** - Contains the non-AI solution using OpenCV and classical computer vision techniques

## AI Solution

The AI solution uses YOLOv5, a deep learning object detection model, to detect and count objects in images.

### Setup

```bash
cd AI
pip install -r requirements.txt
```

### Running the Solution

```bash
python count_objects_yolo.py
```

For more details, please refer to the README.md file in the AI directory.

## Non-AI Solution

The Non-AI solution uses classical computer vision techniques with OpenCV to detect and count objects in images.

### Setup

```bash
cd Non_AI
pip install -r requirements.txt
```

### Running the Solution

```bash
python count_objects.py
```

For more details, please refer to the README.md file in the Non_AI directory.

## Results

Both solutions generate:
- Annotated images with object masks
- Count information displayed on the images
- Console output summarizing the results

## Training Method for AI Solution

The AI solution uses YOLOv5, which was trained using a semi-supervised approach:
1. Initial manual annotation of a subset of images
2. Training YOLOv5 on these annotations
3. Using the trained model to predict on remaining images
4. Manual correction of predictions
5. Retraining the model on the expanded dataset

The model was trained with the following parameters:
- Batch size: 16
- Epochs: 100
- Image size: 640x640
- Learning rate: 0.01

For more detailed information about the training method, please refer to the TRAINING_METHOD.md file in the AI directory. 