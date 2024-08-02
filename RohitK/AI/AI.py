from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO('best.pt')
results = model(source='c:/Users/rohit/Downloads/Counting Stars/img1.jpg', conf=0.5, save=False)


num_objects = len(results[0].boxes)
print(f'Number of objects detected: {num_objects}')


object_detection = results[0].plot()

img = cv2.cvtColor(object_detection, cv2.COLOR_BGR2RGB)


plt.imshow(img)
plt.axis('off')
plt.show()
