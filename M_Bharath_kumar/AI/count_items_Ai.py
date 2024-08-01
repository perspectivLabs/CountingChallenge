import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def detect_objects(img, model):
    results = model(img)
    return results

def draw_labels(img, results, save_path):
    labels, confidences, boxes = results.xyxyn[0][:, -1], results.xyxyn[0][:, -2], results.xyxyn[0][:, :-2]
    img_with_labels = img.copy()
    count = 0
    
    for box, label, confidence in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = int(box[0] * img.shape[1]), int(box[1] * img.shape[0]), int(box[2] * img.shape[1]), int(box[3] * img.shape[0])
        cv2.rectangle(img_with_labels, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_with_labels, f"{model.names[int(label)]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        count += 1
    
    img_with_labels_bgr = cv2.cvtColor(img_with_labels, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_with_labels_bgr)
    return img_with_labels, count

image_path = '/home/bharath/Desktop/ScrewAndBolt_20240713-20240731T040939Z-001/ScrewAndBolt_20240713/20240713_193659.jpg'
output_path = '/home/bharath/CountingChallenge/M_Bharath_kumar/AI/output.jpg'
img = load_image(image_path)
results = detect_objects(img, model)
labeled_img, count = draw_labels(img, results, output_path)

plt.figure(figsize=(10, 10))
plt.imshow(labeled_img)
plt.title(f"Detected Items: {count}")
plt.show()
print(f'Number of items detected: {count}')

