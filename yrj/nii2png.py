import os
import nibabel as nib
import cv2
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

img_dir=r'C:\Users\allstar\nnUNet_raw\Dataset008_HepaticVessel\imagesTr'
mask_dir=r'C:\Users\allstar\nnUNet_raw\Dataset008_HepaticVessel\labelsTr'

save_dir=r'C:\Users\allstar\nnUNet_raw\Dataset008_HepaticVessel\origin'

def get_pixels_hu(scans):
    imgs=np.stack([s.pixel_array for s in scans])
    imgs=imgs.astype(np.int16)
    imgs[imgs==-2000]=0

    intercept=scans[0].RescaleIntercept
    slope=scans[0].RescaleSlope

    if slope !=1:
        imgs=slope*imgs.astype(np.float64)
        imgs=imgs.astype(np.int16)

    imgs+=np.int16(intercept)

    return np.array(imgs,dtype=np.int16)

def windowing(img,hu_min,hu_max,is_normal=False):
    windowWidth=hu_max-hu_min
    newimg=(img-float(hu_min))/float(windowWidth)
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    if not is_normal:
        newimg=(newimg*255).astype('uint8')
    return newimg

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

tot=0
for name in tqdm(os.listdir(img_dir)):
    img_data=nib.load(os.path.join(img_dir,name)).get_fdata()
    img_data=windowing(img_data,-100,500)

    mask_data=nib.load(os.path.join(mask_dir,name)).get_fdata()

    x,y,z=img_data.shape
    tem=tot
    for i in range(z):
        img_slice=img_data[:,:,i]
        img_slice=np.rot90(img_slice)
        img_slice=np.fliplr(img_slice)
        cv2.imwrite(os.path.join(save_dir,f'{tot:05d}.png'),cv2.cvtColor(img_slice.astype(np.uint8),cv2.COLOR_GRAY2RGB))
        tot+=1

    tot=tem
    for i in range(z):
        mask_slice=mask_data[:,:,i]
        mask_slice[mask_slice==1]=255
        mask_slice[mask_slice==2]=0
        mask_slice=np.rot90(mask_slice)
        mask_slice=np.fliplr(mask_slice)
        cv2.imwrite(os.path.join(save_dir,f'{tot:05d}_mask.png'),mask_slice)
        tot+=1

