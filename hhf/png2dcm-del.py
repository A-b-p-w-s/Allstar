"""
测试同济联影截图数据对齐后将png转换成dcm数据的代码

"""
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

    # 读取 PNG 文件并保存为 DICOM 文件
    for i, (ds, _) in enumerate(dicom_data):
        png_file = cv2.imread(os.path.join(png_floder, f"ct_{i+1}.png"))
        png_file = cv2.cvtColor(png_file, cv2.COLOR_BGR2GRAY)
        png_file[png_file>0]=1
    
        # 获取图像的色彩类型和位深（假设是灰度图）
        bits_allocated = ds.BitsAllocated

        # 创建一个新的黑色图像（灰度图像，全黑）
        if bits_allocated == 16:
            black_image = np.array(png_file).astype(np.uint16)
        else:
            black_image = np.array(png_file).astype(np.uint8)
         
        
        # 创建新的 DICOM 数据集
        new_ds = ds.copy()

        # 替换 PixelData为全黑的图像
        new_ds.PixelData = black_image.tobytes()
        
        # 确保其他元数据正确
        new_ds.InstanceNumber = i + 1  # 保证切片的顺序
        # 保存新的黑色切片 DICOM 文件
        output_dcm_path = os.path.join(output_folder, f"IM_{i+1}.dcm")
        new_ds.save_as(output_dcm_path)
        print(f"保存新的 DICOM 文件：{output_dcm_path}")
        
        
# 使用示例
dcm_folder = r'D:\TJ_DATA_CT\DICOM\PA1\ST0\SE1'  # DICOM 文件夹路径
output_folder = r'D:\TJ_data_T\A2_dcm1'  # 输出文件夹路径
png_floder = r'D:\TJ_data_T\fg' #图片

save_dcm_as_png(dcm_folder, output_folder,png_floder)