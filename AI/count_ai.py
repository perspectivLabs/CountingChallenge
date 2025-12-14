import cv2
import numpy as np
import os
import glob

# CONFIGURATION
# Input folder path
INPUT_FOLDER = r"C:\Users\Pranathi Kothapalli\Downloads\ScrewAndBolt_20240713-20251214T035534Z-3-001\ScrewAndBolt_20240713"
OUTPUT_FOLDER = "output_results"

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: The directory '{INPUT_FOLDER}' was not found.")
        print("Please edit the INPUT_FOLDER variable in this script to point to your images.")
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

    print(f"Found {len(image_files)} images. Switch to K-Means ML (CPU Optimized)...")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"Processing: {filename}...")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 1. Preprocessing for K-Means
        # Convert to RGB and reshape to a list of pixels
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixel_values = img_rgb.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # 2. Define criteria and apply K-Means Clustering (Unsupervised ML)
        # We assume 2 main clusters: Background and Objects (or 3 if there's shadow)
        k = 2 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 3. Reconstruct the segmented image
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape((img.shape))

        # 4. Create a binary mask for the objects
        # We assume objects are NOT the dominant cluster (usually background is larger)
        # Calculate frequency of labels to find background (most frequent)
        labels_flat = labels.flatten()
        counts = np.bincount(labels_flat)
        background_label = np.argmax(counts)
        
        # Create mask: 255 where label != background, 0 otherwise
        mask = np.where(labels.reshape(img.shape[:2]) != background_label, 255, 0).astype(np.uint8)

        # 5. Post-process mask (clean noise)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 6. Count connected components (Objects)
        # num_labels will be N+1 (background is 0)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Filter small noise blobs by area
        min_area = 500 # Adjust based on image resolution
        valid_objects = []
        for i in range(1, num_labels): # Skip 0 (background)
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                valid_objects.append(i)
        
        count = len(valid_objects)

        # 7. Visualization
        final_img = img.copy()
        
        # Draw bounding boxes and centroids for valid objects
        for i in valid_objects:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            (cX, cY) = centroids[i]
            
            # Green Box
            cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(final_img, (int(cX), int(cY)), 4, (0, 0, 255), -1)

        print(f"  - K-Means Count: {count}")
        
        # Save output
        save_path = os.path.join(OUTPUT_FOLDER, f"processed_{filename}")
        cv2.imwrite(save_path, final_img)
        print(f"  - Saved to: {save_path}")

    print("\nProcessing Complete.")
    print(f"All results saved in: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()
