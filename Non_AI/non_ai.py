import cv2
import numpy as np
import os

def process_image(image_path, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edged = cv2.Canny(blurred, 30, 150)
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    output_image = image.copy()
    num_items = len(contours)
    num_masks = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        num_masks += 1
    
    # Add text annotation
    cv2.putText(output_image, f"Items Count: {num_items}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output_image, f"Masks Count: {num_masks}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
    
    # Save the result
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
        
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, output_image)

    # Return the number of items detected
    return num_items, num_masks

def process_directory(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            items_count, masks_count = process_image(image_path, output_dir)
            print(f"Processed {filename}: {items_count} items detected, {masks_count} masks drawn")

if __name__ == "__main__":

    screws_and_bolts_dir = 'original_datasets/ScrewAndBolt_20240713'
    screws_and_bolts_output_dir = 'original_datasets/nonAI_output_ScrewAndBolt'
    
    screws_dir = 'original_datasets/Screws_2024_07_15'
    screws_output_dir = 'original_datasets/nonAI_output_Screws'
    
    print("Processing Screws_and_bolts dataset...")
    process_directory(screws_and_bolts_dir, screws_and_bolts_output_dir)
    
    print("Processing screws dataset...")
    process_directory(screws_dir, screws_output_dir)
