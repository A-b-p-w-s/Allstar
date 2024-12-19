import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as F

import os
from tqdm import tqdm
from GRFBUNet import *
# from model import *
from newDataset import *

def dice_coefficient(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def iou_score(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data_dir=r'C:\Users\allstar\nnUNet_raw\Dataset008_HepaticVessel\origin'
# model_dir=r'C:\Users\allstar\Desktop\ves'

data_dir=r'C:\Users\allstar\nnUNet_raw\Dataset008_HepaticVessel\origin'
model_dir=r'C:\Users\Administrator\Desktop\ves'

# 图像增强只用于训练集
class SequentialRotationTransform:
    def __init__(self, angles):
        self.angles = angles
        self.index = 0  # 记录当前使用的角度索引

    def __call__(self, img):
        angle = self.angles[self.index]
        self.index = (self.index + 1) % len(self.angles)  # 依次循环使用角度
        return TF.rotate(img, angle)

# 定义 x 和 y 的 transform
x_transform = transforms.Compose([
    transforms.ToTensor(),
    SequentialRotationTransform([0, 90, 180, 270]),  # 依次旋转 90°, 180°, 270°
    transforms.Normalize([0.5], [0.5])           # 标准化
])

y_transform = transforms.Compose([
    transforms.ToTensor(),
    SequentialRotationTransform([0, 90, 180, 270]),  # 和 x_transform 一致
])

# x_transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5],[0.5])
# ])

# y_transform=transforms.ToTensor()


num_epochs = 1000
bs = 8

writer = SummaryWriter(log_dir='seg_logs\seg')

# images_root = r'C:\Users\allstar\Desktop\ves\imagesTr_dicom'
# labels_root = r'C:\Users\allstar\Desktop\ves\labelsTr_dicom'

images_root = r'C:\Users\Administrator\Desktop\ves\imagesTr_dicom'
labels_root = r'C:\Users\Administrator\Desktop\ves\labelsTr_dicom'
venous_dataset = VesselDataset(
    images_root=images_root,
    labels_root=labels_root,
    transform=x_transform,
    target_transform=y_transform
)

train_size = int(0.8 * len(venous_dataset))
test_size = len(venous_dataset) - train_size

train_dataset, test_dataset = random_split(venous_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

# model = Unet(1, 1).to(device)
model = GRFBUNet(1, 1).to(device)

# model_path = os.path.join(model_dir, 'lastmodel')
# if os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path, map_location='cpu'))
#     print(f"Model loaded from {model_path}")
# else:
#     print(f"No model found at {model_path}, skipping load.")
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters())

bestscore = 0
best_tot = 0

# 用于存储训练过程日志
train_logs = []

for epoch in range(1, num_epochs):
    model.train()
    dt_size = len(train_loader.dataset)
    total = (dt_size // train_loader.batch_size + 1)
    step = 0
    epoch_dice = 0
    epoch_iou = 0
    epoch_loss = 0
    for x, y in tqdm(train_loader,ncols=70):
        step += 1
        inputs = x.to(device)
        labels = y.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        with torch.no_grad():
            dice = dice_coefficient(outputs, labels)
            iou = iou_score(outputs, labels)
            epoch_loss += loss.item()
            epoch_dice += dice
            epoch_iou += iou
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f'epoch:{epoch}/{num_epochs} {step}/{total} loss:{loss.item():0.3f} dice:{dice:0.3f} iou:{iou:0.3f}')

    train_loss = epoch_loss / total
    train_dice = epoch_dice / total
    train_iou = epoch_iou / total

    print(f'  epoch:{epoch}/{num_epochs} mloss:{train_loss:0.3f} mdice:{train_dice:0.3f} miou:{train_iou:0.3f}')

    writer.add_scalar(f'loss/train_loss', train_loss, epoch)
    writer.add_scalar(f'loss/train_dice', train_dice, epoch)
    writer.add_scalar(f'loss/train_iou', train_iou, epoch)

    model.eval()
    epoch_loss = 0
    epoch_dice = 0
    epoch_iou = 0
    dt_size = len(test_loader.dataset)
    total = (dt_size // test_loader.batch_size + 1)
    with torch.no_grad():
        for x, y in test_loader:
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)
            dice = dice_coefficient(outputs, labels)
            iou = iou_score(outputs, labels)
            epoch_dice += dice
            epoch_iou += iou

    test_dice = epoch_dice / total
    test_iou = epoch_iou / total

    writer.add_scalar(f'dice/test_dice', test_dice, epoch)
    writer.add_scalar(f'iou/test_iou', test_iou, epoch)

    print(f'  epoch:{epoch}/{num_epochs} test:     mdice{test_dice:0.3f} miou:{test_iou:0.3f}')

    # 保存日志
    train_logs.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_dice': train_dice,
        'train_iou': train_iou,
        'test_dice': test_dice,
        'test_iou': test_iou
    })
    # 保存日志到 Excel
    df = pd.DataFrame(train_logs)
    excel_path = os.path.join(model_dir, 'training_logs.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Training logs saved to {excel_path}")

    if epoch % 20 == 0:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, 'lastmodel'))
    if train_dice > bestscore:
        best_tot = 0
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, 'bestmodel'))
        bestscore = train_dice
    best_tot += 1
    if best_tot > 50:
        break


