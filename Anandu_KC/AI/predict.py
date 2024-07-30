

from ultralytics import YOLO
import cv2

# Initialize a YOLO-World model
model = YOLO('yolov8m-world.pt')  # or choose yolov8m/l-world.pt

# Define custom classes
model.set_classes(["screw"])

# Execute prediction for specified categories on an image
results = model.predict(source="/home/anandu/task_ai/CountingChallenge/Anandu_KC/AI/img3.jpg", show=False, conf=0.00005, iou=0.5, save=True)

# Count the detected objects
count = len(results[0].boxes.xyxy.cpu())
print(f"Number of detected screws: {count}")

# Load the original image
image_path = "/home/anandu/task_ai/CountingChallenge/Anandu_KC/AI/img3.jpg"
image = cv2.imread(image_path)

# Add text to the image
text = f"Number of detected objects: {count}"
font = cv2.FONT_HERSHEY_SIMPLEX
position = (10, 50)  # position to place the text
font_scale = 1
font_color = (255, 0, 0)  # Blue color in BGR
thickness = 2
line_type = cv2.LINE_AA

# Put the text on the image
cv2.putText(image, text, position, font, font_scale, font_color, thickness, line_type)

# Draw the detected bounding boxes on the image
for box in results[0].boxes.xyxy.cpu().numpy():
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Resize the image to fit the screen
screen_width, screen_height = 1100, 900 # Replace with your screen resolution
scale_factor = min(screen_width / image.shape[1], screen_height / image.shape[0])
new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
resized_image = cv2.resize(image, new_size)

# Show the resized image
cv2.imshow('Detected Objects', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Save the image with the annotations
# cv2.imwrite('/home/anandu/task_ai/Screws_2024_07_15-20240730T124517Z-001/Screws_2024_07_15/img3_detected.jpg', image)
