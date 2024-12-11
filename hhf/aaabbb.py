import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 打开.nii.gz文件
mask_nii_file = r'C:\Users\allstar\Desktop\bbb\aaabbb.nii'
nii_file = r'C:\Users\allstar\Desktop\bbb\boundary.nii'
img = nib.load(nii_file)
mask_img = nib.load(mask_nii_file)
# 获取数据
data = img.get_fdata()
mask = mask_img.get_fdata()

mask = mask+1
mask[mask>1]=0

# 假设数据是3D的，我们有一个轴是切片轴
slice_axis = 2  # 这取决于你的数据，可能需要调整


# windowing 函数
def windowing(img, window_width, window_center):
    minwindow = float(window_center) - 0.5 * float(window_width)
    new_img = (img - minwindow) / float(window_width)
    return new_img

processed_slices = []

X=[]

# 显示每个切片
for i in range(data.shape[slice_axis]): 
    # 读取图像并转换为灰度图
    image = data[:, :,i]
    mask_data = mask[:, :,22]
    
    image = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=5)
    
    image = np.where(mask_data == 0, 0, image)
    if i in [18,19,20,21,22,188,189,190,191]:
        image = cv2.dilate(data[:, :,27], np.ones((3, 3), np.uint8), iterations=5)
        image = np.where(mask_data == 0, 0, image)
        processed_slices.append(image)
    else:    
        processed_slices.append(image)
    
    
# 将处理后的切片堆叠回3D数组
processed_data = np.stack(processed_slices, axis=slice_axis)


# 确保数据类型与原始数据类型相同
processed_data = processed_data.astype(data.dtype)

# 创建一个新的Nifti1Image对象
new_img = nib.Nifti1Image(processed_data, img.affine, img.header)

# 保存处理后的图像
output_file = r'C:\Users\allstar\Desktop\bbb\aaa_boundary.nii.gz'
nib.save(new_img, output_file)