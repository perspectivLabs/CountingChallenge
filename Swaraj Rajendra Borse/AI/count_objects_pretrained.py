import torch
import cv2
import numpy as np
import os
import glob
from pathlib import Path

def detect_objects(model, img_path, conf_threshold=0.25):
    """
    Run inference with pre-trained YOLOv5 model and count objects
    """
    # Run inference
    results = model(img_path)
    
    # Get detection results
    pred = results.xyxy[0].cpu().numpy()  # xmin, ymin, xmax, ymax, confidence, class
    
    # Filter by confidence and relevant classes (small objects)
    # Class indices for small objects in COCO: 
    # 73: laptop, 74: mouse, 75: remote, 76: keyboard, 77: cell phone, etc.
    # Feel free to modify as needed based on your specific requirements
    small_object_classes = [39, 40, 41, 42, 73, 74, 75, 76, 77]  # Bottle, wine glass, cup, fork, laptop, mouse, etc.
    
    # Filter by class and confidence
    filtered_pred = []
    for det in pred:
        cls = int(det[5])
        conf = det[4]
        if conf >= conf_threshold and (cls in small_object_classes):
            filtered_pred.append(det)
    
    filtered_pred = np.array(filtered_pred) if filtered_pred else np.zeros((0, 6))
    
    # Get the original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return 0, None
        
    mask = np.zeros_like(img)
    result_img = img.copy()
    
    # Draw bounding boxes and create mask
    for i, det in enumerate(filtered_pred):
        xmin, ymin, xmax, ymax, conf, cls = det
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
        # Random color for each detection
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        
        # Draw bounding box
        cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Fill mask
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), color, -1)
        
        # Add number label
        cv2.putText(result_img, str(i+1), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 2)
    
    # Combine original image and mask
    alpha = 0.3
    overlay = cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)
    
    # Add count text
    count = len(filtered_pred)
    cv2.putText(overlay, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2)
    
    return count, overlay

def get_objects_non_ai(image_path):
    """
    Use the non-AI method as a backup to refine AI detection results
    Returns the count and potential objects for verification
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try both Otsu and Adaptive thresholding
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3, 3), np.uint8)
    opening1 = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    opening2 = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours for both methods
    contours1, _ = cv2.findContours(opening1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(opening2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use the method that finds more objects
    if len(contours1) >= len(contours2):
        contours = contours1
    else:
        contours = contours2
    
    # Filter contours by area
    min_area = 100
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return len(valid_contours)

def process_directory(model, input_dir, output_dir):
    """
    Process all images in a directory using the model
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    results = []
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        # Detect and count objects using AI
        count_ai, result = detect_objects(model, img_path)
        
        # Get count with Non-AI method for verification
        count_non_ai = get_objects_non_ai(img_path)
        
        # Choose the more reasonable count
        if count_ai == 0 or (count_non_ai > 0 and abs(count_non_ai - count_ai) > count_non_ai * 0.5):
            # If AI detected nothing or the difference is too large, use Non-AI count
            final_count = count_non_ai
            method = "Non-AI (AI failed)"
            # Add corrected count text to the image
            cv2.putText(result, f"Corrected Count: {final_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)
        else:
            final_count = count_ai
            method = "AI"
        
        # Save result
        output_path = os.path.join(output_dir, f"result_{filename}")
        cv2.imwrite(output_path, result)
        
        results.append({
            "filename": filename,
            "count": final_count,
            "method": method
        })
        
        print(f"  Found {final_count} objects using {method} method")
    
    return results

if __name__ == "__main__":
    # Paths
    dataset1 = "../Non_AI/ScrewAndBolt_20240713"
    dataset2 = "../Non_AI/Screws_2024_07_15"
    
    output_dir1 = "yolo_results_ScrewAndBolt"
    output_dir2 = "yolo_results_Screws"
    
    # Load pre-trained YOLOv5 model
    print("Loading pre-trained YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Process images with model
    print("Processing ScrewAndBolt dataset...")
    results1 = process_directory(model, dataset1, output_dir1)
    
    print("\nProcessing Screws dataset...")
    results2 = process_directory(model, dataset2, output_dir2)
    
    print("\nSummary:")
    print("ScrewAndBolt dataset:")
    for result in results1:
        print(f"  {result['filename']}: {result['count']} objects ({result['method']} method)")
    
    print("\nScrews dataset:")
    for result in results2:
        print(f"  {result['filename']}: {result['count']} objects ({result['method']} method)") 