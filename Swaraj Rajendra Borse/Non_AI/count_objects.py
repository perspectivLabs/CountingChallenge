import cv2
import numpy as np
import os
import glob
from pathlib import Path

def count_objects(image_path):
    """
    Count objects in an image using classical computer vision techniques
    Returns the count and the mask image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return 0, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area to remove noise
    min_area = 100  # Adjust based on your images
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Create mask image
    mask = np.zeros_like(img)
    
    # Draw contours on original image and mask
    result_img = img.copy()
    for i, cnt in enumerate(valid_contours):
        # Draw contour with random color
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.drawContours(result_img, [cnt], -1, color, 2)
        cv2.drawContours(mask, [cnt], -1, color, -1)  # Fill the contour
        
        # Add a number label
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(result_img, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2)
    
    # Combine original image and mask with transparency
    alpha = 0.3
    overlay = cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)
    
    # Add count text
    count = len(valid_contours)
    cv2.putText(overlay, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2)
    
    return count, overlay

def adaptive_thresholding_method(image_path):
    """
    Alternative method using adaptive thresholding
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    min_area = 100
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Create mask and result
    mask = np.zeros_like(img)
    result_img = img.copy()
    
    for i, cnt in enumerate(valid_contours):
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.drawContours(result_img, [cnt], -1, color, 2)
        cv2.drawContours(mask, [cnt], -1, color, -1)
        
        # Add number label
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(result_img, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2)
    
    # Combine original image and mask
    alpha = 0.3
    overlay = cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)
    
    # Add count text
    count = len(valid_contours)
    cv2.putText(overlay, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2)
    
    return count, overlay

def process_directory(input_dir, output_dir):
    """
    Process all images in a directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    results = []
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        # Try both methods and use the one with higher count (usually more accurate)
        count1, result1 = count_objects(img_path)
        count2, result2 = adaptive_thresholding_method(img_path)
        
        # Choose the better result (typically the one with higher count is more accurate for this task)
        if count1 >= count2:
            count, result = count1, result1
            method = "Otsu"
        else:
            count, result = count2, result2
            method = "Adaptive"
        
        # Save result
        output_path = os.path.join(output_dir, f"result_{filename}")
        cv2.imwrite(output_path, result)
        
        results.append({
            "filename": filename,
            "count": count,
            "method": method
        })
        
        print(f"  Found {count} objects using {method} method")
    
    return results

if __name__ == "__main__":
    # Process both datasets
    dataset1 = "ScrewAndBolt_20240713"
    dataset2 = "Screws_2024_07_15"
    
    output_dir1 = "results_ScrewAndBolt"
    output_dir2 = "results_Screws"
    
    print("Processing ScrewAndBolt dataset...")
    results1 = process_directory(dataset1, output_dir1)
    
    print("\nProcessing Screws dataset...")
    results2 = process_directory(dataset2, output_dir2)
    
    print("\nSummary:")
    print("ScrewAndBolt dataset:")
    for result in results1:
        print(f"  {result['filename']}: {result['count']} objects ({result['method']} method)")
    
    print("\nScrews dataset:")
    for result in results2:
        print(f"  {result['filename']}: {result['count']} objects ({result['method']} method)") 