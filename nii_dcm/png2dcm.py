import os
import pydicom
from PIL import Image
import numpy as np
import cv2

def save_dcm_as_png(dcm_folder, output_folder,png_floder):
    # 创建输出文件夹，如果不存在的话
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹中所有 .dcm 文件
    dcm_files = os.listdir(dcm_folder)

    # 读取 DICOM 文件并存储切片信息
    dicom_data = []
    for dcm_file in dcm_files:
        dcm_path = os.path.join(dcm_folder, dcm_file)
        ds = pydicom.dcmread(dcm_path)
        
        # 获取 InstanceNumber 或 SliceLocation
        instance_number = None
        if 'InstanceNumber' in ds:
            instance_number = ds.InstanceNumber
        elif 'SliceLocation' in ds:
            instance_number = ds.SliceLocation

        # 如果切片顺序信息存在，添加到 dicom_data 列表中
        if instance_number is not None:
            dicom_data.append((ds, instance_number))

    # 按切片顺序（InstanceNumber 或 SliceLocation）排序
    dicom_data.sort(key=lambda x: x[1])

    # 遍历排序后的 DICOM 文件并保存为 PNG
    for i, (ds, _) in enumerate(dicom_data):
        png_file = cv2.imread(os.path.join(png_floder, f"ct_{i+1}.png"))
        
        rows = ds.Rows
        columns = ds.Columns
        # 获取图像的色彩类型和位深（假设是灰度图）
        bits_allocated = ds.BitsAllocated

        # 创建一个新的黑色图像（灰度图像，全黑）
        if bits_allocated == 16:
            black_image = np.zeros((rows, columns), dtype=np.uint16)  # 16位图像
        else:
            black_image = np.zeros((rows, columns), dtype=np.uint8)  # 8位图像

        # 创建新的 DICOM 数据集
        new_ds = ds.copy()

        # 替换 PixelData为全黑的图像
        new_ds.PixelData = black_image.tobytes()

        # 确保其他元数据正确
        new_ds.InstanceNumber = i + 1  # 保证切片的顺序

        # 保存新的黑色切片 DICOM 文件
        output_dcm_path = os.path.join(output_folder, f"black_slice_{i+1}.dcm")
        new_ds.save_as(output_dcm_path)
        print(f"保存新的 DICOM 文件：{output_dcm_path}")
        
        
# 使用示例
dcm_folder = r'C:\Users\allstar\Desktop\aaaa\DICOM (2)\nii\SE5'  # DICOM 文件夹路径
output_folder = r'C:\Users\allstar\Desktop\aaaa\new_dcm'  # 输出文件夹路径
png_floder = r'C:\Users\allstar\Desktop\aaaa\dicom_png'

save_dcm_as_png(dcm_folder, output_folder,png_floder,png_floder)