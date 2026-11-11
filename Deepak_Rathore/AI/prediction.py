
import os
import time
from pathlib import Path
from collections import Counter
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_PATH = Path("v3_best.pt")

IMAGE_SIZE = 1024
CONFIDENCE = 0.60
IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 400

# =========================================================
# 2. LOAD SEGMENTATION MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH.resolve()}"
    )

print("Loading YOLO11 segmentation checkpoint...")

model = YOLO(str(MODEL_PATH))

if model.task != "segment":
    raise ValueError(
        f"Expected segmentation model, got: {model.task}"
    )

print("Model loaded successfully!")
print("Task:", model.task)
print("Classes:", model.names)

# =========================================================
# 3. SELECT IMAGE FROM LOCAL COMPUTER
# =========================================================

def get_local_image_path():

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")
        ]
    )

    root.destroy()
    return file_path


image_path = get_local_image_path()

if not image_path:
    print("No image selected.")
    raise SystemExit

filename = os.path.basename(image_path)

print("\nSelected image:", filename)

# =========================================================
# 4. RUN PREDICTION
# =========================================================

start_time = time.perf_counter()

results = model.predict(
    source=image_path,
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE,
    iou=IOU_THRESHOLD,
    max_det=MAX_DETECTIONS,
    save=False,
    verbose=False
)

inference_time = time.perf_counter() - start_time

result = results[0]

# =========================================================
# 5. COUNT OBJECTS AND COLLECT CONFIDENCE
# =========================================================

counts = Counter()
class_confidences = {
    "Nuts": [],
    "Screws": []
}

if result.boxes is not None:

    for cls_id, confidence in zip(
        result.boxes.cls.int().tolist(),
        result.boxes.conf.tolist()
    ):

        class_name = result.names[cls_id]

        counts[class_name] += 1

        if class_name in class_confidences:
            class_confidences[class_name].append(
                float(confidence)
            )

# =========================================================
# 6. DISPLAY SEGMENTATION RESULT
# =========================================================

annotated_image = result.plot()

# Ultralytics returns BGR; Matplotlib expects RGB.
annotated_image = annotated_image[:, :, ::-1]

plt.figure(figsize=(14, 10))
plt.imshow(annotated_image)
plt.title(f"YOLO11 Segmentation — {filename}")
plt.axis("off")
plt.tight_layout()
plt.show()

# =========================================================
# 7. PRINT RESULTS
# =========================================================

nuts = counts["Nuts"]
screws = counts["Screws"]

print("\n" + "=" * 45)
print("           PREDICTION REPORT")
print("=" * 45)

print("Image:", filename)
print("Confidence threshold:", CONFIDENCE)
print("IoU threshold:", IOU_THRESHOLD)

print("-" * 45)

print("Nuts detected:", nuts)
print("Screws detected:", screws)
print("Total objects:", nuts + screws)

print("-" * 45)

for class_name, values in class_confidences.items():

    if values:
        average_confidence = sum(values) / len(values)

        print(
            f"{class_name} average confidence: "
            f"{average_confidence:.2%}"
        )

print("-" * 45)

print(f"Inference time: {inference_time * 1000:.2f} ms")

print("=" * 45)
