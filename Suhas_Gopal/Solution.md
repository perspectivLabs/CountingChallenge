# Counting Challenge Solution


## Setup

Clone the repository:
```bash
git clone https://github.com/Suhas-G/CountingChallenge.git
```

Create a conda/virtual env and install Pytorch according to your system specifications from [Pytorch](https://pytorch.org/get-started/locally/).  
For additional dependencies:

```bash
cd Suhas_Gopal
pip install -r requirements.txt
```

## Non AI

In Non AI section, I have used only morphological operations, hough transforms and contour detection to detect the individual objects.

### Assumptions:
1. Given image has darker foreground compared to background, allowing automatic thresholding.
2. The structure of nuts are very different from other screws, and also shows tighter packing (so most of them are visually connected). Using same algorithm for both gave pretty bad results. So, I have assumed that images with nuts in them (eg. ScrewAndBolt_20240713\20240713_194630.jpg) are known beforehand, and I have 2 different techniques - one for screws/bolts and another for nuts.
3. The objects are quite far away from image edges - this is used in removing vignette.


### Method
1. Convert given image to binary using OTSU automatic thresholding.  
2. Some of the images have a vignette - A hacky way to remove this.  
    a. Find the contours in the binary image.  
    b. Vignette happens at edges / corners of the image.  
    c. Remove all the contours that are touching the edge of image.  
3. Process this image based on whether it is screw image or a nuts image.  
    - Screws:  
        a. Dilate the binary image slightly to smoothen out sharp parts of the image.  
        b. Do distance transform for the image and do thresholding on this, to get only the peaks  (the centre patches for each object ideally).  
        c. Find the contours of this image, to count the no of objects, and also draw bounding boxes.  
    - Nuts:  
        a. From step 2, the contours are filtered to keep only the inner contours, using the hierarchy information.  
        b. These contours are passed to circle detection using Hough transforms. The idea is to find the holes at the centre of each nut.  
        c. The circles are drawn on the image and counted.  

### Instructions to run

```bash
cd Suhas_Gopal/Non_AI/
python count.py -i <data folder> -o <output folder>
```
`<data folder>` - Folder containing the images. Eg. data/ScrewAndBolt_20240713.  
`output folder>` - Path to folder where the output images and count results will be stored.

### Results


| Image filename | count |
|----------------|-------|
| img1.jpg | 43 |
| img1_43_nosy.jpg | 43 |
| img2.jpg | 43 |
| img3.jpg | 43 |
| img4.jpg | 43 |
| img5.jpg | 43 |
| img6.jpg | 43 |
| 20240713_192951.jpg | 21 |
| 20240713_193135.jpg | 21 |
| 20240713_193650.jpg | 9 |
| 20240713_193659.jpg | 9 |
| 20240713_193831.jpg | 185 |
| 20240713_193839.jpg | 196 |
| 20240713_193907.jpg | 183 |
| 20240713_194200.jpg | 88 |
| 20240713_194206.jpg | 85 |
| 20240713_194215.jpg | 92 |
| 20240713_194232.jpg | 90 |
| 20240713_194256.jpg | 84 |
| 20240713_194316.jpg | 82 |
| 20240713_194541.jpg | 353 |
| 20240713_194551.jpg | 349 |
| 20240713_194606.jpg | 340 |
| 20240713_194621.jpg | 322 |
| 20240713_194630.jpg | 334 |


## AI

In AI section, I tried using GroundingDino tiny and YOLOWorld for zero-shot detection, but they gave unreliable results - prompt 'single_metal_part' worked better for some images, while 'small_part' worked better for some images (specific classes like 'screws', 'bolts' didn't seem to do well), but even then it could not detect correctly in images having too many objects, even at very low confidence thresholds (1e-3/1e-4). So I decided to fine tune a model for these objects. Since the given dataset is small and manual annotation is very time consuming, I used an external dataset - [MVTec Screws](https://www.kaggle.com/datasets/ipythonx/mvtec-screws). It has all the kinds of objects with minor differences, though images here are having very different background, and also the resolution is smaller.

Considering the limitations of time and compute, I chose Yolov8-nano. I converted the dataset to yolo format and trained the model for 100 epochs.
It can recognise most objects, but still cannot reach accuracy of 95% (by visual inspection of results) and produces duplicate detections.



### Instructions to run

```bash
cd Suhas_Gopal/Non_AI/
python predict.py -i <data folder> -o <output folder>
```
`<data folder>` - Folder containing the images. Eg. data/ScrewAndBolt_20240713.  
`output folder>` - Path to folder where the output images and count results will be stored.


### Results

| Image filename | count |
|----------------|-------|
| img1.jpg | 43 |
| img1_43_nosy.jpg | 43 |
| img2.jpg | 43 |
| img3.jpg | 43 |
| img4.jpg | 43 |
| img5.jpg | 43 |
| img6.jpg | 43 |
| 20240713_192951.jpg | 23 |
| 20240713_193135.jpg | 22 |
| 20240713_193650.jpg | 10 |
| 20240713_193659.jpg | 9 |
| 20240713_193831.jpg | 180 |
| 20240713_193839.jpg | 171 |
| 20240713_193907.jpg | 177 |
| 20240713_194200.jpg | 74 |
| 20240713_194206.jpg | 74 |
| 20240713_194215.jpg | 84 |
| 20240713_194232.jpg | 76 |
| 20240713_194256.jpg | 79 |
| 20240713_194316.jpg | 76 |
| 20240713_194541.jpg | 364 |
| 20240713_194551.jpg | 362 |
| 20240713_194606.jpg | 358 |
| 20240713_194621.jpg | 361 |
| 20240713_194630.jpg | 359 |


### Pending
I started the implementation for tiling the high resolution image to lower resolution, performing detection on it and then merging the duplicate detections. But as of now, this did not improve the results. So more experiments are needed.  
I only used YOLOv8 Nano for fine-tuning, there are better models that can handle smaller objects (or even bigger versions of YOLOv8), which I will try to use next.