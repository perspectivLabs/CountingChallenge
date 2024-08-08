## Task 1 : Counting and detecting using non-AI technique ##
The steps used in exeuting the project are as follows:
 * Read the image through opencv
 * Converting the image to grayscale
 * Blurring the image to reomove the noise in the image. Applying the best guassian blur on the image.
 * Convertingt the image to binary image. where pixels with intensity above 60 are set to 0 and and pixels below 60 are set to be 255 
 * The next step is the morphological opertions to clear the small holes in the binary image.
 * The next step is to find the contours in the binary image.
 * Filetring the countours based on the size.
 * Draw contours on original image and the count the contours.
## Task 2 : Counting and detecting based on AI techniques.
 The steps involved in detecting the items in a image and putting  bounding box using AI.
  * The first thing is to gather the data and then annotate the items in the image using a software called roboflow.
  * The annotation images are added to the dataset and then exported to the required format.
  * Since I decided to use YOLOv8 for the detection and counting i decided to export in YOLOV8 format.
  * Next step is to export the dataset and the start training it on YOLOv8 nano model because it's a small model and takes a very little time for training.
  * The next thing after training is to run a test image and count the number of boxes as it gives the number of items in the image.
  * After training the result of the test image are displayed.

## Both techniques yield a better results if the training dataset has more images. Also it is not a probelm with the low number of dataset as we can train this on small models and then detect the items. ##
