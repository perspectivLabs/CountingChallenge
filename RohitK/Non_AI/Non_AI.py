import cv2
import matplotlib.pyplot as plt


image = cv2.imread('c:/Users/rohit/Downloads/Counting Stars/img1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 170)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))


contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

print(f'Number of objects detected: {len(contours)}')


plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
