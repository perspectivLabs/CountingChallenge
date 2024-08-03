import cv2
import numpy as np
class cv2algo():
    def infer(self,input_image_path):
        image = cv2.imread(input_image_path)
        "Convert to grayscale"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        "Apply GaussianBlur to reduce noise and improve contour detection"
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        "Apply adaptive thresholding"
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
        "Apply morphological operations to improve object separation"
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
        "Find contours"
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        "Create an empty image for masks"
        masked_image = np.zeros_like(image)
        "Draw contours and count objects"
        object_count = 0
        for contour in contours:
            "Calculate the area of the contour"
            area = cv2.contourArea(contour)
            if area > 20:  # Adjust the area threshold as needed
                object_count += 1
                "Create a mask"
                mask = np.zeros(image.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [contour], -1, 255, -1)
                "Apply the mask to the original image"
                single_masked = cv2.bitwise_and(image, image, mask=mask)               
                "Combine the single masked image with the overall masked_image"
                masked_image = cv2.add(masked_image, single_masked)                
                "Draw the contour on the original image"
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        print(f'Number of objects detected: {object_count}')
        return object_count,image
        # plt.figure()
        # plt.title(input_image_path)
        # plt.imshow(image)
        # plt.show()