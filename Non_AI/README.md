# Non-AI Object Counting Solution

## Method Overview
This solution uses classical Computer Vision techniques provided by OpenCV to detect, segment, and count objects without deep learning models.

### Approaches Used:
1. **Preprocessing**:
   - **Grayscale Conversion**: Simplifies the image to a single channel.
   - **Gaussian Blur**: Reduces high-frequency noise to improve thresholding.

2. **Segmentation**:
   - **Thresholding**: adaptive or Otsu's thresholding is used to separate objects from the background.
   - **Morphological Operations**: Closing and Opening are applied to fill gaps in contours and remove small noise specs.

3. **Counting**:
   - **Contour Detection**: `cv2.findContours` identifies separate object boundaries.
   - **Area Filtering**: Small contours (noise) are ignored based on a minimum area threshold.
   - **Visualization**: Valid contours are drawn on the image, and a count is displayed.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```bash
   jupyter notebook counting_opencv.ipynb
   ```
3. Update the `IMAGE_PATH` variable in the notebook to point to your test image.
4. Run all cells to see the count and visualization.

## Accuracy Calculation
Accuracy is calculated as:
\[ \text{Accuracy} = \left( 1 - \frac{| \text{Predicted Count} - \text{Ground Truth} |}{\text{Ground Truth}} \right) \times 100\% \]
*Note: Ground truth must be provided manually for verification.*
