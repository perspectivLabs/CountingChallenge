import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import os

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self.conv_block(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        b = self.bottleneck(p2)
        d2 = self.up2(b); d2 = torch.cat((d2, e2), dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat((d1, e1), dim=1); d1 = self.dec1(d1)
        return torch.sigmoid(self.final(d1))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Fixed LR
    criterion = nn.BCELoss()
    
    train_dir = "/home/aipilot/AI/dataset/Train"
    img_list = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
    
    print(f"Training on {len(img_list)} images...")
    for epoch in range(20): # 20 epochs for better convergence
        model.train()
        total_loss = 0
        for name in img_list:
            img = cv2.imread(os.path.join(train_dir, name))
            img_in = cv2.resize(img, (256, 256))
            
            # Create Pseudo-Label
            gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
            _, label = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Tensors
            x = torch.from_numpy(img_in).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
            y = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(img_list):.4f}")

    torch.save(model.state_dict(), "/home/ai/AI/unet_weights.pth")
    print("Training complete. Weights saved.")

if __name__ == "__main__":
    train()
