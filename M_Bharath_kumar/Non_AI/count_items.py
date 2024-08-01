import cv2
import numpy as np

image_path = '/home/bharath/Desktop/Screws_2024_07_15-20240731T040939Z-001/Screws_2024_07_15/img3.jpg'
print(f'Loading image from: {image_path}')

image = cv2.imread(image_path)
if image is None:
    print('Failed to load image. Please check the file path and integrity.')
else:
    print('Image loaded successfully.')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('Converted to grayscale successfully.')

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print('Applied Gaussian blur successfully.')

    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    print('Applied adaptive thresholding successfully.')

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=2)
    print('Applied morphological operations successfully.')

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f'Found {len(contours)} contours.')

    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    print('Contours drawn on image.')

    num_items = len(contours)
    print(f'Number of items: {num_items}')

    cv2.imwrite('Non_AI/result.png', image)
    print('Result image saved as Non_AI/result.png.')

