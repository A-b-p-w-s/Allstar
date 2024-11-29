import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib as plt
# Import the UNet model (assuming it's defined in model.py)
from model import UNet

# Function to calculate Dice coefficient
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

# Function to calculate Dice coefficient for 3 classes
def dice_coef_3Class(y_true, y_pred, numLabels=3):
    dice = 0
    for index in range(numLabels):
        dice += dice_coef(y_true[:, index, :, :], y_pred[:, index, :, :])
    return dice / numLabels

# Custom dataset class for liver CT scans
class CustomLiverCTDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load and preprocess image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Normalize if needed (example normalization, adjust as per your need)
        image = image.astype(np.float32) / 255.0

        # Resize image if needed (example resizing, adjust as per your need)
        image = cv2.resize(image, (512, 512))

        # Convert image to tensor and rearrange dimensions
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)

        # Load and preprocess mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = torch.tensor(mask, dtype=torch.long)  # Convert mask to LongTensor

        return image, mask


# Function to load data paths
def load_data(img_dir, mask_dir):
    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))
    
    img_paths = [os.path.join(img_dir, img) for img in images]
    mask_paths = [os.path.join(mask_dir, mask) for mask in masks]

    return img_paths, mask_paths

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define paths to your dataset
    train_frame_path = 'D:\\Liver_tarek\\train_images'
    train_mask_path = 'D:\\Liver_tarek\\train_masks'
    
    # Load data paths
    train_x, train_y = load_data(train_frame_path, train_mask_path)
    
    # Define hyperparameters
    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    IMG_CHANNELS = 3
    n_classes = 3
    lr = 1e-4
    batch_size = 4
    epochs = 50
    
    # Create datasets and dataloaders
    train_dataset = CustomLiverCTDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = UNet(n_classes=n_classes, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, img_channels=IMG_CHANNELS).to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # Use Cross Entropy Loss for multi-class segmentation
    
    # Training loop
    history = {'loss': []}

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Ensure targets are within the range [0, n_classes-1]
            targets = targets.squeeze(1)  # Remove channel dimension for CrossEntropyLoss
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())

        
        epoch_loss = np.mean(epoch_losses)
        history['loss'].append(epoch_loss)
        
        # Print epoch statistics
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), 'Final_model_Multiclass-Semantic_U-NET.pth')
    
    # Convert history to DataFrame and save as CSV
    hist_df = pd.DataFrame(history)
    hist_csv_file = 'history.csv'
    hist_df.to_csv(hist_csv_file, index=False)
    
    # Plot training history
    # Since transformations were removed, only plot the loss
    plt.figure(figsize=(12, 6))
    plt.plot(history['loss'], linewidth=3)
    plt.ylabel("Loss", fontsize=20)
    plt.xlabel("Epoch", fontsize=20)
    plt.legend(["loss"], loc="upper right", fontsize=15)
    plt.show()
