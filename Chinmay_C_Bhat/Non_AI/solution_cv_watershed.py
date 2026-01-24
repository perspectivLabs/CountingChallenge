import cv2
import numpy as np
import os
import argparse
import glob

def count_items_in_image(image_path, debug_out=True):
    # 1. Load and Pre-process
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read the image: {image_path}")
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # 2. Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Clean noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. Distance Transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # FIX: Increase threshold to avoid double-counting
    _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 5. Unknown Region
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Marker Labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 7. Apply Watershed
    markers = cv2.watershed(img, markers)

    # 8. Extract and Count Results
    obj_ids = np.unique(markers)
    count = 0
    output = img.copy()

    for obj_id in obj_ids:
        if obj_id <= 1: 
            continue

        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == obj_id] = 255

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 500: 
                count += 1
                if debug_out:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
                    cv2.putText(output, f"#{count}", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if debug_out:
        debug_path = image_path.replace(".jpg", "_watershed.jpg")
        cv2.imwrite(debug_path, output)
        print(f"Saved debug output to {debug_path}")

    return count

if __name__ == "__main__":
    # Example usage
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else r"c:\pr\perspectiv\dataset\ScrewAndBolt_20240713\20240713_192951.jpg"
    cnt = count_items_in_image(path)
    print(f"Count: {cnt}")
