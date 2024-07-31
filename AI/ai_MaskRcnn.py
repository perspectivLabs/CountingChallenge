import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
import os
from PIL import Image

def load_model():
    # pre-trained Mask R-CNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = model.to('cuda') if torch.cuda.is_available() else model
    model.eval()
    return model

def process_image(model, image_path, output_dir):
    # Read the image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((800, 800)) # Resizing to a smaller size
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # Convert image to OpenCV format
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    num_items = len(predictions["boxes"])
    num_masks = len(predictions["masks"])  # Number of masks generated

    # Draw bounding boxes and masks
    for i in range(len(predictions["boxes"])):
        box = predictions["boxes"][i].cpu().numpy().astype(int)
        score = predictions["scores"][i].item()
        label = f"Object: {score:.2f}"
        
        if score > 0.5:  # Confidence threshold
            cv2.rectangle(image_cv, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image_cv, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw mask
            mask = predictions["masks"][i, 0].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8) * 255
            color_mask = np.zeros_like(image_cv)
            color_mask[:, :, 1] = mask
            image_cv = cv2.addWeighted(image_cv, 1.0, color_mask, 0.5, 0)

    cv2.putText(image_cv, f"Items Count: {num_items}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_cv, f"Masks Count: {num_masks}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    # Save the result in the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image_cv)

    return num_items, num_masks

def process_directory(model, input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            items_count, masks_count = process_image(model, image_path, output_dir)
            print(f"Processed {filename}: {items_count} items detected, {masks_count} masks drawn.")

if __name__ == "__main__":
    
    model = load_model()

    screws_and_bolts_dir = 'Original_datasets/ScrewAndBolt_20240713'
    screws_and_bolts_output_dir = 'Original_datasets/mask_AI_output_ScrewAndBolt'

    screws_dir = 'Original_datasets/Screws_2024_07_15'
    screws_output_dir = 'Original_datasets/mask_AI_output_Screws'

    print("Processing Screws_and_bolts dataset...")
    process_directory(model, screws_and_bolts_dir, screws_and_bolts_output_dir)

    print("Processing screws dataset...")
    process_directory(model, screws_dir, screws_output_dir)
