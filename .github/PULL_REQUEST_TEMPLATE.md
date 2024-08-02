## Info

### Name

Kunal Singh Rajpurohit

### Python Version

Python 3.10.4 or google colab

## Description

The count.ipynb file is present in both AI and Non_AI folder containing the solution as per direction

### AI

The solution utilizes a Faster R-CNN model with a ResNet-50 backbone, provided by the PyTorch library, to perform object detection on a dataset of images. Here’s a summary of how the solution was derived:

1) Model Loading:

A pre-trained Faster R-CNN model is loaded using PyTorch’s torchvision library. This model, which uses a ResNet-50 backbone, is adept at detecting objects within images.

2) Image Processing:

Images are loaded and converted into RGB format using the PIL library.
These images are then transformed into tensors, which are required for processing by the model.

3) Object Detection:

The model is used to predict bounding boxes and associated scores (confidence levels) for detected objects in the images. The torch.no_grad() context is used to avoid computing gradients, which speeds up inference and reduces memory usage.

4) Drawing Results:

Detected objects are visualized by drawing bounding boxes around them and annotating them with confidence scores using OpenCV.
The total count of detected items and available masks is displayed on the image.

5) Saving Results:

The processed images, which now include bounding boxes and annotations, are saved to a specified output directory. The output directory is created if it does not already exist.
Processing Multiple Images:

The solution iterates through all JPEG images in the specified input directory, applying the detection model to each image and saving the results.

### Non_AI

The solution outlined employs classical image processing techniques to detect and count screws in images without using machine learning. Here’s a summary of how the solution was derived:

1) Image Loading:

Images are read using OpenCV's cv2.imread function.
Grayscale Conversion:

The images are converted to grayscale using cv2.cvtColor. This simplifies the image and reduces the complexity of further processing.

2) Image Blurring:

Gaussian Blur is applied to the grayscale image to reduce noise and smooth out the image. This helps in better thresholding and contour detection.

3) Thresholding:

Adaptive thresholding is used to segment the image into binary form. This step helps in distinguishing objects (screws) from the background by converting the image into a binary format where objects of interest are highlighted.

4)Morphological Operations:

Morphological operations, specifically closing (using cv2.morphologyEx), are applied to fill in small gaps and remove noise, enhancing the contours of objects in the image.

5) Contour Detection:

Contours are detected using cv2.findContours, which identifies the boundaries of objects in the binary image.

6) Contour Filtering and Bounding Boxes:

Contours are filtered based on their area to remove small, irrelevant contours. For each valid contour, a bounding box is drawn around it to highlight detected screws.

7) Results Display:

The processed images, which include bounding boxes around detected screws, are displayed using Matplotlib. Each image is shown with the count of detected screws as part of the title.

8) Output:

The count of screws for each image is printed to the console.
This approach leverages traditional image processing techniques such as filtering, thresholding, and contour detection to identify and count objects in images, which is suitable for applications where machine learning might be too complex or not necessary.

## Additional Comments

The Non_AI solution is giving pretty good results when compared to AI part, since AI part requires lot of research on finding the best architecture to use and continously testing with different parameters