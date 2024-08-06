## Info

### Name
Atul Priyank

### Python Version
Above Python 3.9.0

## Description

### AI
The AI-based solution for detecting and counting screws and bolts in images was derived using two state-of-the-art models: GroundingDINO and RetinaNet, both integrated with the Detectron2 framework. GroundingDINO was utilized for object detection, leveraging its robust capabilities to process images and identify objects based on a given text prompt. The model was fine-tuned to detect small objects like screws and bolts with high accuracy.

The pipeline involved:

1. Image Processing with GroundingDINO: This model was used to detect objects in images by processing them with a text prompt, which instructed the model to focus on finding small screws and bolts.
2. Post-Processing: The outputs from GroundingDINO were post-processed to filter and scale the detected bounding boxes.
3. Training RetinaNet with Detectron2: The annotated images were used to train the RetinaNet model, a powerful object detection architecture. Detectron2 was employed for training and evaluation, providing a robust framework for managing datasets, training models, and evaluating performance.

### Non_AI
The Non-AI solution for detecting and counting screws and bolts utilized traditional image processing techniques, specifically the Sobel Edge Detector, combined with contour detection.

1. Image Preprocessing: Images were converted to grayscale and the Sobel edge detector was applied to highlight edges, helping identify the boundaries of screws and bolts.
2. Edge Detection with Sobel Operator: Sobel operators in both horizontal and vertical directions were used to detect edges. The resulting edge maps were combined to form a magnitude image.
3. Thresholding and Contour Detection: The edge-detected image was thresholded to create a binary image. Contours were then detected and counted to determine the number of screws and bolts.
4. Visualization: Detected contours were overlaid on the original image, and the resulting images were saved to visualize the detection results.


## Additional Comments
The AI-based approach leverages advanced deep learning models, which are highly accurate but require significant computational resources and data for training. This method is suitable for large-scale projects where high accuracy is crucial.

The Non-AI approach is simpler and more computationally efficient, relying on classical image processing techniques. It is effective for basic object detection tasks, especially when the objects have well-defined shapes and sizes, making it suitable for scenarios with limited resources or simpler datasets.
