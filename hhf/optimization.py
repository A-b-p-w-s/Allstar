"""
优化三维重建数据的代码
边缘检测提取切片外轮廓减少重建的点

"""

import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
import cv2

input_file = r"C:\Users\allstar\Desktop\aaaaa\tumor-213\ct_213_0000.nii.gz"
output_file = r'C:\Users\allstar\Desktop\aaaaa\tumor-213\optimization.nii.gz'

boundary_img=nib.load(input_file)
nii_data = boundary_img.get_fdata()
print(np.sum(nii_data))

array= []

for i in range(nii_data.shape[2]):
    # 提取特定切片并转换为灰度图
    gray = (nii_data[:, :, i] * 255).astype(np.uint8)
    
    
    edges = cv2.Canny(gray, 100, 200)/255

 
    array.append(edges.astype(int))
 
array2 = []   
for i in range(nii_data.shape[0]):
    # 提取特定切片并转换为灰度图
    gray = (nii_data[i, :, :] * 255).astype(np.uint8)
    
    
    edges = cv2.Canny(gray, 100, 200)/255
    
    array2.append(edges.astype(int))

 
array2 = np.array(array2)
array = np.array(array)

array2 = array2.transpose((2, 0, 1))
 
array = array + array2

array[array>0] = 1

# 确保数据类型与原始数据类型相同
array=array.astype(nii_data.dtype)

# 将处理后的切片堆叠回3D数组
processed_data = np.stack(array, axis=2)


# 创建一个新的Nifti1Image对象
new_img = nib.Nifti1Image(processed_data, boundary_img.affine, boundary_img.header)

# 保存处理后的图像
nib.save(new_img, output_file)
print(np.sum(array))