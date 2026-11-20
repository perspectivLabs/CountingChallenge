# AI technique

I am provided a very small dataset which of just 25 images in total. Training on it is meaning less because of overfitting and it is just wrong deciding to train on such a small dataset. Therefore i chose to go for zero shot testing with pretrained segmentation models. I tried yolo segmentation, yoloe and SAM 2.1 models.

NOTE: I tested my entire solution in the GPU machines i have available to me in my office.

## Zero shot testing
Yolo segmentation did not have class labels for screw and nuts since those classes are not present in the COCO dataset. So, I compared the results of SAM 2.1 and YoloE. 

YoloE was pretty bad even when i used the 'B' size model. 

The SAM 2.1 model was very impressive, it gave the most accurate results out of any results i have seen including the results i got from Non_AI techniques. Those results will be good enough for the challenge but that is a very sub optimal solution for one major reason: ** Compute requirement **

I have been the lead of machine learning operation in my current position in my current company and i just can't ignore the cost of compute at scale. The SAM 2.1 took nearly 20 GB in VRAM and also was the slowest in terms of inference speed.  So i had to develop a new solution which would satisfy me in both results and compute requirements.

## Training a new yolo segmentation model
Dataset of 25 images is out of question for training the model since not only is it too small but also it is my test dataset.

### Synthetic dataset
I decided to make a synthetic dataset by extracting assets from the test dataset (25 images) using SAM 2 and creating my own dataset by randomly adding screws and nuts to a set of randomly selected backgrounds.

Non of the images have a plain white background. So i created gray background, linear gradient illumination to mimic uneven lighting and ring gradient to simulate the vignette in background to use as background for new training images

I got over 600 assets by segmenting out the screws and the nuts (made sure that the centre ring of the nuts have been cut out) and arranged them in a 640 x 640 image with varying density by controlling the count of objects withing each image. Range for each image is randomly set between 25 and 400 for each image (count of items in each image). Suppose if items overlap then the masking will be done on only the visible part of the item. The final dataset consist of around 6000 images

I am sharing the drive link of my synthetic dataset: https://drive.google.com/drive/folders/1HyzB1Ekloy8oE1EqwyM-lc8QvxlBdn6r?usp=sharing

### Training
I chose yolo segmentation model (I prefer yolo for their efficiency compared to it's model size and results and also how easy it is to use it with Nvidia Deepstream). Trained the yolo seg 'medium' model for 50 epochs using the ultralytics library. Ultralytics stopped the training at 33rd epoch since the best result was achieved in 17th epoch and no improvements was seen afterwards.

### Results

The yolo model is not only better at counting the objects but also it is on average 100x faster than SAM 2.1 . 

Results with comparison grids to reference image and output of sam and yolo segmentation models is here: https://drive.google.com/drive/folders/1BZGHweJowkiKn9-vtFR5blQq-2l95UuL?usp=drive_link
