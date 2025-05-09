AI-Based Counting Challenge Solution (Siddesh)
YOLOv8 Segmentation is used to count items in an image and overlay segmentation masks.
Steps: Run item_counter_yolov8. Place the img1, img2, img3 files in the main folder for detection or update the path.(Place input image in the main folder and update the path in the code. Run the .ipynb code )

Presently, the YOLOv8 segmentation model (`yolov8n-seg.pt`) is being used. To achieve the desired accuracy mentioned in the challenge a custom annotated dataset needs to be created & used for training and then images of bolts and screws can be run on. Run inference on input image, count number of detected items, show image with segmentation overlays.

Requirements: contains all the needed packages to run