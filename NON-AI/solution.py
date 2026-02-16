import cv2
import numpy as np
import matplotlib.pyplot as plt

def solve_non_ai(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find image at {image_path}")
        return
    
    # 2. Pre-processing: Grayscale and Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Thresholding (Otsu's Binarization)
    # Note: Use cv2.THRESH_BINARY if items are lighter than background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Remove small noise with Morphological Opening
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 5. Watershed to separate touching objects
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area (Distance Transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply Watershed
    markers = cv2.watershed(img, markers)
    
    # 6. Generate Masks and Count
    output_img = img.copy()
    count = 0
    unique_markers = np.unique(markers)
    
    for marker in unique_markers:
        if marker <= 1: # Skip background and boundaries
            continue
        
        # Create a mask for this specific item
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == marker] = 255
        
        # Generate random color for overlay
        color = np.random.randint(0, 255, (3,)).tolist()
        
        # Overlay the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output_img, contours, -1, color, -1)
        count += 1

    # 7. Save and Display Results
    cv2.putText(output_img, f"Count: {count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    cv2.imwrite('Non_AI/output_mask.png', output_img)
    print(f"Process complete. Items detected: {count}")
    
    # Visual check (Matplotlib is safer than cv2.imshow for Ubuntu servers)
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    solve_non_ai('/home/aipilot/Non_AI/image/20240713_193659.jpg') # Ensure your image is named input.jpg
