import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image, cmap='gray'):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

def count_screws(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary thresholding (invert the result to get white screws on black background)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Display the binary thresholded image
    display_image('Binary Thresholded Image', thresh)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and count the contours based on area
    screw_count = 0
    min_area = 5500 # Adjust this threshold based on the size of your screws
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            screw_count += 1
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Draw contours on the original image
            # Display the contour area for debugging
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(image, str(area), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save and display the mask overlay image
    cv2.imwrite('Non_AI/mask_overlay.png', image)
    display_image('Detected Screws with Contours', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return screw_count

# Example usage
image_path = '/home/anandu/task_ai/CountingChallenge/Anandu_KC/Non_AI/20240713_192951.jpg'  # Update the path to the actual image
screw_count = count_screws(image_path)
print(f"Number of screws detected: {screw_count}")
