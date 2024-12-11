import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import numpy as np
import random

from torch.utils.data import DataLoader,random_split,Subset
from torchvision import datasets,transforms
from torch.utils.tensorboard import SummaryWriter

import os

from model import *
# from dataset import *
from newDataset import *

def dice_coefficient(pred,target,smooth=1e-5):
    pred=pred.view(-1)
    target=target.view(-1)
    intersection=(pred*target).sum()
    dice=(2.0*intersection+smooth)/(pred.sum()+target.sum()+smooth)
    return dice

def iou_score(pred,target,smooth=1e-5):
    pred=pred.view(-1)
    target=target.view(-1)
    intersection=(pred*target).sum()
    union=pred.sum()+target.sum()-intersection
    iou=(intersection+smooth)/(union+smooth)
    return iou.item()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

set_seed(42)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir=r'C:\Users\allstar\nnUNet_raw\Dataset008_HepaticVessel\origin'
model_dir=r'C:\Users\allstar\nnUNet_raw'

x_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
])

y_transform=transforms.ToTensor()

num_epochs=1000
bs=8

writer=SummaryWriter(log_dir='seg_logs\seg')

# venous_dataset=VesselDataset(data_dir,transform=x_transform,target_transform=y_transform)
images_root = r'C:\Users\allstar\nnUNet_raw\Dataset008_HepaticVessel\imagesTr_dicom'
labels_root = r'C:\Users\allstar\nnUNet_raw\Dataset008_HepaticVessel\labelsTr_dicom'
venous_dataset = VesselDataset(
    images_root=images_root,
    labels_root=labels_root,
    transform=x_transform,
    target_transform=y_transform
)
train_size=int(0.8*len(venous_dataset))
test_size=len(venous_dataset)-train_size

train_dataset,test_dataset=random_split(venous_dataset,[train_size,test_size])

train_loader=DataLoader(train_dataset,batch_size=bs,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=bs,shuffle=False)

model = Unet(1,1).to(device)
criterion=torch.nn.BCELoss()
optimizer=optim.Adam(model.parameters())

bestscore=0
best_tot=0
for epoch in range(1,num_epochs):
    model.train()
    dt_size=len(train_loader.dataset)
    total=(dt_size//train_loader.batch_size+1)
    step=0
    epoch_dice=0
    epoch_iou=0
    epoch_loss=0
    for x,y in train_loader:
        step+=1
        inputs=x.to(device)
        labels=y.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        with torch.no_grad():
            dice=dice_coefficient(outputs,labels)
            iou=iou_score(outputs,labels)
            epoch_loss+=loss.item()
            epoch_dice+=dice
            epoch_iou+=iou
        loss.backward()
        optimizer.step()
        if step%50==0:
            print(f'epoch:{epoch}/{num_epochs} {step}/{total} loss:{loss.item():0.3f} dice:{dice:0.3f} iou:{iou:0.3f}')

    print(f'  epoch:{epoch}/{num_epochs} mloss:{epoch_loss/(total):0.3f} mdice:{epoch_dice/(total):0.3f} miou:{epoch_iou/(total):0.3f}')

    writer.add_scalar(f'loss/train_loss',epoch_loss/(total),epoch)
    writer.add_scalar(f'loss/train_dice', epoch_dice / (total), epoch)
    writer.add_scalar(f'loss/train_iou', epoch_iou / (total), epoch)

    model.eval()
    epoch_loss=0
    epoch_dice=0
    epoch_iou=0
    dt_size=len(test_loader.dataset)
    total=(dt_size//test_loader.batch_size+1)
    with torch.no_grad():
        for x,y in test_loader:
            inputs=x.to(device)
            labels=y.to(device)
            outputs=model(inputs)
            dice=dice_coefficient(outputs,labels)
            iou=iou_score(outputs,labels)
            epoch_dice+=dice
            epoch_iou+=iou

        writer.add_scalar(f'dice/test_dice',epoch_dice/(total),epoch)
        writer.add_scalar(f'iou/test_iou',epoch_iou/(total),epoch)
    print(f'  epoch:{epoch}/{num_epochs} test:     mdice{epoch_dice/(total):0.3f} miou:{epoch_iou/(total):0.3f}')
    if epoch%20==0:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(),os.path.join(model_dir,'lastmodel'))
    if epoch_dice/(total)>bestscore:
        best_tot=0
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(),os.path.join(model_dir,'bestmodel'))
        bestscore=epoch_dice/(total)
    best_tot+=1
    if(best_tot>50):
        break