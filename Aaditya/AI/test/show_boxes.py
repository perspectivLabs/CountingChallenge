from ultralytics import YOLO
import cv2
import os

# Load the trained model with confidence threshold and IoU settings
model = YOLO('/Users/aaditya/Desktop/PESPECTIV/runs/detect/train/weights/best.pt')
model.conf = 0.25  # Confidence threshold (0-1)
model.iou = 0.45   # NMS IoU threshold (0-1)

test_folder = 'Screws_2024_07_15'

for image_name in os.listdir(test_folder):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(test_folder, image_name)
        image = cv2.imread(image_path)
        results = model(image)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                cv2.rectangle(image, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                label = f'screw {confidence:.2f}'
                cv2.putText(image, label, 
                          (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (0, 255, 0), 2)
        cv2.imshow('Detections', image)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break

cv2.destroyAllWindows()