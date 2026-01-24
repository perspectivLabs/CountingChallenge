import cv2
import numpy as np
import os
import glob

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all jpg images
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images to process.")

    for path in image_paths:
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to read {path}")
            continue

        # 1. Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding often works better for uneven lighting
        # But for these simple screws on white/plain BG, simple threshold might work.
        # Let's try Otsu's binarization after blurring
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Invert if background is light and objects are dark (usually the case for screws on white)
        # Check average brightness to decide inversion logic if needed, but standard is usually BG=Light

        # Binary Inverse Threshold
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Morphological operations
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # 3. Contour Detection
        contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Filter Contours
        min_area = 100 # Adjust this threshold based on image resolution
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        count = len(valid_contours)
        print(f"{filename}: Found {count} items")

        # 5. Visualize and Save
        # Draw Contours
        result_img = img.copy()
        cv2.drawContours(result_img, valid_contours, -1, (0, 255, 0), 2)

        # Overlay Mask (Red tint)
        mask = np.zeros_like(img)
        cv2.drawContours(mask, valid_contours, -1, (0, 0, 255), -1)

        overlay = cv2.addWeighted(result_img, 0.7, mask, 0.3, 0)

        # Put text count
        cv2.putText(overlay, f"Count: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, overlay)

if __name__ == "__main__":
    # Assuming dataset structure
    dataset_screw_bolt = r"c:\pr\perspectiv\dataset\ScrewAndBolt_20240713"
    dataset_screws = r"c:\pr\perspectiv\dataset\Screws_2024_07_15"

    output_base = r"c:\pr\perspectiv\Chinmay_Bhat\Non_AI"

    print("Processing ScrewAndBolt Dataset...")
    process_images(dataset_screw_bolt, os.path.join(output_base, "output_screw_bolt"))

    print("\nProcessing Screws Dataset...")
    process_images(dataset_screws, os.path.join(output_base, "output_screws"))
