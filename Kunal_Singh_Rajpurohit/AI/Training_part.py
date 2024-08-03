import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        img_tensor = TF.to_tensor(img)
        
        # Dummy target (replace with actual annotations)
        target = {}
        target["boxes"] = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)  # Example
        target["labels"] = torch.tensor([1], dtype=torch.int64)  # Example
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img_tensor, target

    def __len__(self):
        return len(self.image_files)

def train_model(num_epochs=10):
    # Load a pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Modify the classifier head to match the number of classes
    in_features = model.roi_heads.box_predictor.in_features
    num_classes = 2  # Background + your custom class
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # DataLoader
    dataset = CustomDataset(image_dir='ScreAndBolt_20240713')
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Optimizer and Loss Function
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training Loop
    model.train()
    for epoch in range(num_epochs):
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item()}")

    # Save the trained model
    torch.save(model.state_dict(), 'fasterrcnn_model.pth')

if __name__ == "__main__":
    train_model()
