import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,transforms
import numpy as np
# from model import *
from GRFBUNet import *
from torch.utils.tensorboard import SummaryWriter
import os
import random
import PIL.Image as Image
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm

def windowing(img,hu_min,hu_max,is_normal=False):
    windowWidth=hu_max-hu_min
    newimg=(img-float(hu_min))/float(windowWidth)
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    if not is_normal:
        newimg=(newimg*255).astype('uint8')
    return newimg

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_file=1

# nii_img_dir=r'C:\Users\allstar\Desktop\data'
# nii_img_name=f'CTA_00{num_file}_0000.nii.gz'

nii_img_dir = r'C:\Users\allstar\Downloads\Task08_HepaticVessel\Task08_HepaticVessel\imagesTr'
# nii_img_dir=r'C:\Users\allstar\Desktop\data'
# nii_img_name=f'CTA_001_0000.nii.gz'
nii_img_name=f'hepaticvessel_001.nii.gz'

save_dir=fr'C:\Users\allstar\Desktop\data\predict'

load_model_root=fr'C:\Users\allstar\Desktop\yrj\unet'
load_model_name=fr'grfb_bestmodel_86'

x_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

y_transform=transforms.ToTensor()

model = GRFBUNet(1,1).to(device)
model.load_state_dict(torch.load(os.path.join(load_model_root,load_model_name),map_location='cpu'))

model.eval()

nii_file=nib.load(os.path.join(nii_img_dir,nii_img_name))
img_data=nii_file.get_fdata(dtype=np.float32) # float32
after_data=np.zeros_like(img_data)

# img_data=windowing(img_data,-100,500)
x,y,z=img_data.shape

for i in tqdm(range(z)):
    img_slice=img_data[:,:,i]
    # img_slice=np.rot90(img_slice)
    # img_slice=np.fliplr(img_slice)
    # img=cv2.cvtColor(img_slice,cv2.COLOR_BGR2RGB)

    x=img_slice
    x=x_transform(x).unsqueeze(0).to(device)

    y=model(x).detach().squeeze(0).cpu().numpy()
    y[y>=0.5]=1
    y[y<0.5]=0
    y=y.squeeze()
    # y=np.fliplr(y)
    # y=np.rot90(y,3)
    after_data[:,:,i]=y

modified_nii=nib.Nifti1Image(after_data, nii_file.affine, nii_file.header)
nib.save(modified_nii,os.path.join(save_dir,nii_img_name))





