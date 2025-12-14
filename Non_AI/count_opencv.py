import cv2
import numpy as np
import os
import glob

# CONFIGURATION
INPUT_FOLDER = r"C:\Users\Pranathi Kothapalli\Downloads\ScrewAndBolt_20240713-20251214T035534Z-3-001\ScrewAndBolt_20240713"
OUTPUT_FOLDER = "output_results"

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: The directory '{INPUT_FOLDER}' was not found.")
        return

    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    if not image_files:
        print(f"No images found in {INPUT_FOLDER}")
        return
        
    print(f"Found {len(image_files)} images. Processing...")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"Processing: {filename}...")
        
        # 1. Load Image
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"  - Error loading {filename}")
            continue

        # 2. Preprocessing
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)

        # 3. Thresholding & Morphology
        thresh_val, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)
        processed_img = opening

        # 4. Count and Visualize
        contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        MIN_AREA = 51
        valid_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_AREA:
                valid_contours.append(cnt)
                
        count = len(valid_contours)
        
        # Draw results
        result_img = original_img.copy()
        cv2.drawContours(result_img, valid_contours, -1, (0, 255, 0), 2)
        
        # Overlay mask
        mask = np.zeros_like(original_img)
        cv2.drawContours(mask, valid_contours, -1, (0, 0, 255), cv2.FILLED)
        
        # Alpha blend
        final_output = cv2.addWeighted(result_img, 1, mask, 0.4, 0)
        
        # Save output
        save_path = os.path.join(OUTPUT_FOLDER, f"processed_{filename}")
        cv2.imwrite(save_path, final_output)
        print(f"  - Objects Detected: {count} | Saved to: {save_path}")

    print("\nProcessing Complete.")
    print(f"All results saved in: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()
