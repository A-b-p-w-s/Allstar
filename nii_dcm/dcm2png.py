import os
import pydicom
from PIL import Image
import numpy as np
from tqdm import tqdm

def convert_dicom_to_png(dicom_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for filename in tqdm(os.listdir(dicom_path)):
        if filename.endswith(".dcm"):
            # 读取DICOM文件
            dicom_file = os.path.join(dicom_path, filename)
            ds = pydicom.dcmread(dicom_file)
            
            # 获取原始像素数据
            img_array = ds.pixel_array
            
            img_array[img_array>2429] = 2429
            
            # 获取DICOM图像的窗宽和窗位，用于归一化
            max_value = np.max(img_array)
            min_value = np.min(img_array)
            
            window_level = (max_value+min_value)/2
            window_width = abs(max_value-window_level)
            
            
            # 归一化到0-255范围
            img_array_normalized = np.clip((img_array - window_level) / (window_width / 255), 0, 255)
            
            # 将归一化后的数组转换为8位无符号整数
            img_array_normalized = img_array_normalized.astype(np.uint8)
            
            # 将numpy数组转换为PIL图像
            img = Image.fromarray(img_array_normalized)
            
            # 构建输出文件的路径
            output_file = os.path.join(output_folder, filename.replace('.dcm', '.png'))
            
            # 保存为PNG格式
            img.save(output_file)

# 设置DICOM文件的文件夹路径和输出文件夹路径
dicom_folder_path = r'C:\Users\allstar\Desktop\new_tm\ct_dicom\0'
output_folder_path = r'C:\Users\allstar\Desktop\new_tm\ct_dicom\png0'

# 调用函数进行转换
convert_dicom_to_png(dicom_folder_path, output_folder_path)