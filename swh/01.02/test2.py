import nibabel as nib
import numpy as np

# 读取 NIfTI 文件
file_path = "E:/data/CTA/A0/CTA_001_0000.nii.gz"
nii_img = nib.load(file_path)

# 获取图像数据
image_data = nii_img.get_fdata()

# 获取图像的形状（维度）
image_shape = image_data.shape
print(f"Image Shape: {image_shape}")

# 获取体素大小
voxel_sizes = nii_img.header.get_zooms()
print(f"Voxel Sizes: {voxel_sizes}")

# 获取仿射变换矩阵
affine_matrix = nii_img.affine
print(f"Affine Matrix: \n{affine_matrix}")

# 获取其他元数据
data_type = nii_img.header.get_data_dtype()
print(f"Data Type: {data_type}")

# 示例：访问特定的体素数据
# 假设你想访问 (x, y, z) 位置的体素数据
x, y, z = 100, 150, 50
voxel_value = image_data[x, y, z]
print(f"Voxel Value at ({x}, {y}, {z}): {voxel_value}")

# 示例：访问整个切片数据
# 假设你想访问第 50 层的切片数据
slice_index = 50
slice_data = image_data[:, :, slice_index]
print(f"Slice Data at index {slice_index}: \n{slice_data}")

# 可视化切片数据（可选）
import matplotlib.pyplot as plt

plt.imshow(slice_data, cmap='gray')
plt.title(f'Slice at index {slice_index}')
plt.colorbar()
plt.show()