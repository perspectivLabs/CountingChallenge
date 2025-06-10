import torch
import cv2
import numpy as np
import os
import glob
from pathlib import Path
import yaml
import shutil
from PIL import Image

def prepare_dataset(src_dir1, src_dir2, dest_dir, split_ratio=0.8):
    """
    Prepare dataset for YOLOv5 training
    """
    # Create directories with absolute paths
    dest_dir_abs = os.path.abspath(dest_dir)
    os.makedirs(os.path.join(dest_dir_abs, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir_abs, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir_abs, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir_abs, 'labels/val'), exist_ok=True)
    
    # Get all image files with absolute paths
    images1 = glob.glob(os.path.join(os.path.abspath(src_dir1), '*.jpg'))
    images2 = glob.glob(os.path.join(os.path.abspath(src_dir2), '*.jpg'))
    all_images = images1 + images2
    
    # Shuffle and split
    np.random.shuffle(all_images)
    split_idx = int(len(all_images) * split_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Create labels (we'll do this manually since we don't have ground truth)
    # This is a simplified approach - in a real scenario, you'd annotate the dataset
    
    # Define dataset.yaml with absolute paths
    dataset_yaml = {
        'path': dest_dir_abs,
        'train': os.path.join(dest_dir_abs, 'images/train'),
        'val': os.path.join(dest_dir_abs, 'images/val'),
        'nc': 1,  # Number of classes
        'names': ['screw']  # Class names
    }
    
    # Write the yaml file
    yaml_path = os.path.join(dest_dir_abs, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    return train_images, val_images, yaml_path

def detect_objects_non_ai(image_path):
    """
    Use non-AI method to generate pseudo-labels for training
    Returns bounding boxes and the count
    """
    img = cv2.imread(image_path)
    if img is None:
        return [], 0
    
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
    
    # Get bounding boxes in YOLO format (normalized x_center, y_center, width, height)
    h, w = img.shape[:2]
    boxes = []
    
    for cnt in valid_contours:
        x, y, box_w, box_h = cv2.boundingRect(cnt)
        
        # Convert to YOLO format
        x_center = (x + box_w / 2) / w
        y_center = (y + box_h / 2) / h
        norm_width = box_w / w
        norm_height = box_h / h
        
        # Class 0 for screw/bolt
        boxes.append([0, x_center, y_center, norm_width, norm_height])
    
    return boxes, len(boxes)

def create_yolo_labels(image_paths, dest_dir):
    """
    Create YOLO format labels for each image
    """
    for img_path in image_paths:
        # Get bounding boxes using non-AI method
        boxes, count = detect_objects_non_ai(img_path)
        
        # Get destination paths
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        
        # Determine if it's train or val
        if img_path in train_images:
            img_dest = os.path.join(dest_dir, 'images/train', filename)
            label_dest = os.path.join(dest_dir, 'labels/train', f"{basename}.txt")
        else:
            img_dest = os.path.join(dest_dir, 'images/val', filename)
            label_dest = os.path.join(dest_dir, 'labels/val', f"{basename}.txt")
        
        # Copy image
        shutil.copy(img_path, img_dest)
        
        # Write label file
        with open(label_dest, 'w') as f:
            for box in boxes:
                f.write(' '.join(map(str, box)) + '\n')
        
        print(f"Processed {filename} - Found {count} objects")

def train_yolo_model(dataset_dir, yaml_path, epochs=50, batch_size=16, img_size=640):
    """
    Train YOLOv5 model
    """
    # Clone YOLOv5 if not already present
    if not os.path.exists('yolov5'):
        os.system('git clone https://github.com/ultralytics/yolov5')
        os.system('pip install -r yolov5/requirements.txt')
    
    # Train YOLOv5s on custom dataset using absolute path
    train_cmd = f"cd yolov5 && python train.py --img {img_size} --batch {batch_size} --epochs {epochs} --data {yaml_path} --weights yolov5s.pt"
    
    print("Starting YOLOv5 training with command:")
    print(train_cmd)
    os.system(train_cmd)
    
    # Get the path to the trained weights
    weights_path = os.path.join(os.path.abspath('yolov5'), 'runs', 'train', 'exp', 'weights', 'best.pt')
    return weights_path

def detect_and_count(model, img_path, conf_threshold=0.25):
    """
    Run inference with YOLOv5 and count objects
    """
    # Run inference
    results = model(img_path)
    
    # Get detection results
    pred = results.xyxy[0].cpu().numpy()  # xmin, ymin, xmax, ymax, confidence, class
    
    # Filter by confidence
    pred = pred[pred[:, 4] >= conf_threshold]
    
    # Get the original image
    img = cv2.imread(img_path)
    mask = np.zeros_like(img)
    result_img = img.copy()
    
    # Draw bounding boxes and create mask
    for i, det in enumerate(pred):
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
    count = len(pred)
    cv2.putText(overlay, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2)
    
    return count, overlay

def process_directory_with_yolo(model, input_dir, output_dir):
    """
    Process all images in a directory using YOLOv5
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    results = []
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        # Detect and count objects
        count, result = detect_and_count(model, img_path)
        
        # Save result
        output_path = os.path.join(output_dir, f"result_{filename}")
        cv2.imwrite(output_path, result)
        
        results.append({
            "filename": filename,
            "count": count
        })
        
        print(f"  Found {count} objects")
    
    return results

if __name__ == "__main__":
    # Paths
    dataset1 = "ScrewAndBolt_20240713"
    dataset2 = "Screws_2024_07_15"
    dataset_dir = "dataset_yolo"
    
    # Create and prepare dataset
    print("Preparing dataset...")
    train_images, val_images, yaml_path = prepare_dataset(dataset1, dataset2, dataset_dir)
    
    # Create YOLO format labels
    print("Creating YOLO labels...")
    create_yolo_labels(train_images + val_images, dataset_dir)
    
    # Train YOLOv5 model
    print("Training YOLOv5 model...")
    weights_path = train_yolo_model(dataset_dir, yaml_path)
    
    # Load trained model
    print(f"Loading model from {weights_path}...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    
    # Process images with trained model
    output_dir1 = "yolo_results_ScrewAndBolt"
    output_dir2 = "yolo_results_Screws"
    
    print("Processing ScrewAndBolt dataset...")
    results1 = process_directory_with_yolo(model, dataset1, output_dir1)
    
    print("\nProcessing Screws dataset...")
    results2 = process_directory_with_yolo(model, dataset2, output_dir2)
    
    print("\nSummary:")
    print("ScrewAndBolt dataset:")
    for result in results1:
        print(f"  {result['filename']}: {result['count']} objects")
    
    print("\nScrews dataset:")
    for result in results2:
        print(f"  {result['filename']}: {result['count']} objects") 