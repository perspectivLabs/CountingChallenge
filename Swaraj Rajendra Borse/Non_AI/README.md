# Non-AI Object Counting Solution

This solution uses traditional computer vision techniques to count objects in images and overlay masks on them.

## Approach

The solution uses two different methods and picks the better result:

1. **Otsu's Thresholding Method**:
   - Convert to grayscale
   - Apply Gaussian blur to reduce noise
   - Apply Otsu's thresholding
   - Find and filter contours
   - Draw contours and masks

2. **Adaptive Thresholding Method**:
   - Convert to grayscale
   - Apply Gaussian blur
   - Apply adaptive thresholding
   - Apply morphological operations
   - Find and filter contours
   - Draw contours and masks

Both methods use contour detection to identify individual objects, apply color masks, and show the count.

## Requirements

```
opencv-python==4.8.0
numpy==1.24.3
```

## How to Run

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the script:
   ```
   python count_objects.py
   ```

The script will process all images in both the ScrewAndBolt and Screws datasets and save the results in the `results_ScrewAndBolt` and `results_Screws` directories, respectively.

## Output

- The script will create annotated images showing:
  - Object outlines with unique colors
  - A colored mask for each object
  - Total count displayed in the top-left corner

- Results will be printed to the console, showing the count for each image.

## Approach Justification

This solution doesn't use AI techniques, relying instead on classical computer vision approaches. The combination of two methods (Otsu's and Adaptive thresholding) ensures better accuracy across different lighting conditions and image qualities. 