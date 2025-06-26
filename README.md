# 🔧 Screw and Bolt Detection Project  
**Author:** Akshat Singh  
**Objective:** Detect and count the number of **screws** and **bolts** in images using both AI and Non-AI methods with >95% accuracy.

---

## 📌 Problem Statement

The goal of this project is to develop two separate systems:

1. 🔍 **Non-AI Solution** using classical image processing (OpenCV)
2. 🤖 **AI Solution** using object detection (YOLOv8 + Roboflow)

Each system should detect, count, and visualize screw and bolt positions in static images.

---


---

## 🧠 AI-Based Approach

### 🚀 Tools Used:
- [Roboflow](https://roboflow.com) – For dataset labeling and augmentation
- [YOLOv8 (Ultralytics)](https://docs.ultralytics.com) – For object detection
- OpenCV, Matplotlib – For image loading and annotation

### 📦 Dataset:
- Custom screw and bolt dataset
- Annotated and exported from Roboflow in YOLOv8 format

### 🧪 Model Training:
- Trained using YOLOv8 with 50 epochs
- Custom classes:
  - `0`: Screw
  - `1`: Bolt

### 🧾 Output:
- Annotated image with bounding boxes
- Console output for detected class, confidence, and total count


---

## 🛠️ Non-AI (OpenCV) Approach

### 🧰 Tools Used:
- OpenCV for grayscale conversion, thresholding, and contour detection
- NumPy, Matplotlib for numerical operations and visualization

### ⚙️ Steps:
- Preprocess the image (blur, threshold, morphology)
- Detect contours
- Filter based on contour area
- Estimate screw/bolt count using median contour area logic

### 📊 Output:
- Annotated image with contours
- Histogram of object areas
- Console output of estimated item count


---

## 📦 Dependencies

Install all required Python libraries via:

```bash
pip install -r requirements.txt

