import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO

def remove_vignette(image):
    rows, cols = image.shape[:2]
    # Create an elliptical mask
    mask = np.zeros((rows, cols), dtype=np.uint8)
    center = (cols // 2, rows // 2)
    axes = (int(cols * 0.6), int(rows * 0.6))  # 1.5 times bigger
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Apply the mask to each channel of the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def count_contours_non_ai(image_path, debug=True, debug_dir="debug"):
    # Read the image
    image = cv2.imread(image_path)

    # Resize the image to 3000x4000
    resized_image = cv2.resize(image, (3000, 4000))

    # Convert the resized image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to binarize the image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_image = remove_vignette(binary_image)

    # Find all contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(binary_image, contours, -1, (0, 0, 0), 20)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    filtered_contours = []
    min_area = 400
    max_area = 25000

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:
            filtered_contours.append(contour)
    
    contours = filtered_contours

    if debug:
        cv2.drawContours(resized_image, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{os.path.basename(image_path)}_output_non_ai.png"), resized_image)

    return len(contours)

def count_contours_ai(image_path, model, debug=False, debug_dir="debug"):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (3000, 4000))
    results = model(resized_image, conf=0.0001, iou=0.5)
    
    # Get bounding boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()

    min_area = 1000
    max_area = 80000
    
    valid_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        area = (x2 - x1) * (y2 - y1)
        if min_area <= area <= max_area:
            valid_boxes.append((x1, y1, x2, y2))
    
    # Count detected objects within the area range
    count = len(valid_boxes)
    
    if debug:
        
        # Draw bounding boxes without text or confidence scores
        for x1, y1, x2, y2 in valid_boxes:
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Save the annotated image
        cv2.imwrite(os.path.join(debug_dir, f"{os.path.basename(image_path)}_output_ai.png"), resized_image)
    
    return count

def main():
    parser = argparse.ArgumentParser(description="Count contours in images using AI or non-AI methods.")
    parser.add_argument("--ai", action="store_true", help="Use AI mode for counting")
    parser.add_argument("--non_ai", action="store_true", help="Use non-AI mode for counting")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--data", type=str, required=True, help="Path to the data directory")
    args = parser.parse_args()

    debug_dir = "debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    if not (args.ai or args.non_ai):
        args.non_ai = True
    
    if args.ai:
        model = YOLO("yolov8m-world.pt")
        model.set_classes(["small_parts"])
    
    for root, _, files in os.walk(args.data):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                
                if args.non_ai:
                    count = count_contours_non_ai(image_path, debug=args.debug)
                    output_file = "output_non_ai.txt"
                elif args.ai:
                    count = count_contours_ai(image_path, model, debug=args.debug)
                    output_file = "output_ai.txt"
                
                with open(output_file, "a") as f:
                    f.write(f"{file}: {count}\n")
                
                print(f"Processed {file}: {count} contours")

if __name__ == "__main__":
    main()