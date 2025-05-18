import cv2
import numpy as np
import os

kernel = np.ones((20,20),np.uint8)
kernel_2 = np.ones((5,5),np.uint8)

folder_path = 'Screws_2024_07_15'
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

for image_file in image_files:
    img_path = os.path.join(folder_path, image_file)
    img = cv2.imread(img_path, 1)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gaussian_blur = cv2.GaussianBlur(denoised, (0, 0), 3.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian_blur, -0.5, 0)
    img_grey = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_grey,128,170)
    closing = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
    erosion = cv2.erode(closing,kernel_2,iterations=2)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of screws {image_file}: {len(contours)}")
    
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Original', img)
    cv2.imshow('Processed', closing)
    cv2.imshow('Contours', result)
    
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
