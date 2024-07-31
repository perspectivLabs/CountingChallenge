import cv2
import os
import numpy as np
from ultralytics import YOLO

def load_model(model_path):
    # Load the trained yolov5 model
    model = YOLO("yolov5s.pt")
    return model

def process_image(model, image_path, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    
    # Run inference on the image
    results = model(image)

    # Extract the number of items detected
    num_items = len(results[0].boxes)
    num_masks = num_items  # Each bounding box corresponds to an item

    # Draw bounding boxes and labels
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = box.conf[0]
        class_id = box.cls[0]
        label = f"{model.names[int(class_id)]} {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

   # Add text annotation for item and mask count
    cv2.putText(image, f"Items Count: {num_items}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Masks Count: {num_masks}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
    
    # Save the result in the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
    
    return num_items, num_masks

def process_directory(model, input_dir, output_dir):    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            items_count, masks_count = process_image(model, image_path, output_dir)
            print(f"Processed {filename}: {items_count} items detected, {masks_count} masks drawn.")

if __name__ == "__main__":
    # Load the trained YOLOv5 model
    model_path = "screw_bolt_yolov5.pt"  # Replace with your trained model path
    model = load_model(model_path)

    screws_and_bolts_dir = 'Original_datasets/ScrewAndBolt_20240713'
    screws_and_bolts_output_dir = 'Original_datasets/AI_v5_output_ScrewAndBolt'

    screws_dir = 'Original_datasets/Screws_2024_07_15'
    screws_output_dir = 'Original_datasets/AI_v5_output_Screws'

    print("Processing Screws_and_bolts dataset...")
    process_directory(model, screws_and_bolts_dir, screws_and_bolts_output_dir)

    print("Processing screws dataset...")
    process_directory(model, screws_dir, screws_output_dir)