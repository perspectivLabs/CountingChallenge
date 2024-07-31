import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import cv2
import numpy as np
import os

# Pre-trained Mask R-CNN model
def load_model():
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model

# Function to process a single image
def process_image(model, image_path, output_dir):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    # Process predictions
    masks = predictions[0]['masks'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    
    threshold = 0.5
    count = 0
    mask_overlay = np.zeros_like(image_rgb)
    
    for i in range(len(masks)):
        if scores[i] > threshold:
            count += 1
            mask = masks[i][0]
            mask_overlay[mask > threshold] = (0, 255, 0)
    
    # Add text annotation for item and mask count
    image_with_annotations = image.copy()
    cv2.putText(image_with_annotations, f"Items Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_with_annotations, f"Masks Count: {count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    # Combine mask overlay with original image
    combined_image = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)

    # Save the result in the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, combined_image)

    return count

def process_directory(model, input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            item_count = process_image(model, image_path, output_dir)
            print(f"Processed {filename}: {item_count} items detected.")

if __name__ == "__main__":
   
    model = load_model()

    screws_and_bolts_dir = 'Original_datasets/ScrewAndBolt_20240713'
    screws_and_bolts_output_dir = 'Original_datasets/newmask_AI_output_ScrewAndBolt'

    screws_dir = 'Original_datasets/Screws_2024_07_15'
    screws_output_dir = 'Original_datasets/newmask_AI_output_Screws'

    print("Processing Screws_and_bolts dataset...")
    process_directory(model, screws_and_bolts_dir, screws_and_bolts_output_dir)

    print("Processing screws dataset...")
    process_directory(model, screws_dir, screws_output_dir)
