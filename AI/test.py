import torch
import torch.nn as nn
import cv2
import numpy as np
import os

# --- 1. ARCHITECTURE (Must match train.py) ---
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

# --- 2. BATCH INFERENCE ENGINE ---
def run_batch_test(test_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet().to(device)
    
    # Load weights with security flag to suppress the warning
    weights_path = "/home/aipilot/AI/unet_weights.pth"
    if not os.path.exists(weights_path):
        print("Error: Weights not found!")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(test_images)} images in {test_dir}")

    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        h, w = img.shape[:2]
        img_in = cv2.resize(img, (256, 256))
        x = torch.from_numpy(img_in).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            pred = model(x).squeeze().cpu().numpy()
        
        mask = cv2.resize(pred, (w, h))
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtering and Overlay
        final_count = 0
        overlay = img.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) > 50: # Area filter
                final_count += 1
                cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), -1)
                cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), 2)

        # Save result with the count in the filename
        save_path = os.path.join(output_dir, f"res_{final_count}_{img_name}")
        cv2.imwrite(save_path, overlay)
        print(f"Processed {img_name}: Detected {final_count} items")

if __name__ == "__main__":
    TEST_DIR = "/home/ai/AI/dataset/Test"
    OUTPUT_DIR = "/home/ai/AI/results"
    
    run_batch_test(TEST_DIR, OUTPUT_DIR)
