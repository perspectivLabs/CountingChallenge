## Info

### Name
Satyaki Bhattacharjee

### Python Version
Python 3.11

## Description

### AI
Used YOLO V8 object dedtection AI/ML model. The ipynb is from Kaggle as I needed CUDA support for GPU Training.

- First I used a data annotator tool to annotate the images of screws for training. The annotations are stored in "data.yaml"
- Then I downloaded YOLO V8 from Ultralytics library. Trained it on the annotated images. Saved the best weights.
- Then I used those weights to predict and count the number of screws.

The number of images for training was low, and there were too many objects to annotate for a single person. More images with 10 - 15 objects per image would significantly increase the trained model's performance. 

### Non_AI
 Used opencv's cv2.inRange() function to create a mask. Used contours to find number of items. Not accurate but this is the best I could do.



## Additional Comments

Thank you for this task. I really appreciate the challenge. I gained significant knowledge through this assignment.