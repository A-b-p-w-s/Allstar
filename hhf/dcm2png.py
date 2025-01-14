"""
dcm数据转换成png的代码    
用于将dcm数据转换成png数据
"""
# import os
# import pydicom
# from PIL import Image
# import numpy as np
# from tqdm import tqdm

# def convert_dicom_to_png(dicom_path, output_folder):
#     # 确保输出文件夹存在
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     # 遍历文件夹中的所有文件
#     for filename in tqdm(os.listdir(dicom_path)):
#         # if filename.endswith(".dcm"):
#         # 读取DICOM文件
#         dicom_file = os.path.join(dicom_path, filename)
#         ds = pydicom.dcmread(dicom_file)
        
#         # 获取原始像素数据
#         img_array = ds.pixel_array
        
#         img_array[img_array>2429] = 2429
        
#         # 获取DICOM图像的窗宽和窗位，用于归一化
#         max_value = np.max(img_array)
#         min_value = np.min(img_array)
        
#         window_level = (max_value+min_value)/2
#         window_width = abs(max_value-window_level)
        
        
#         # 归一化到0-255范围
#         img_array_normalized = np.clip((img_array - window_level) / (window_width / 255), 0, 255)
        
#         # 将归一化后的数组转换为8位无符号整数
#         img_array_normalized = img_array_normalized.astype(np.uint8)
        
#         # 将numpy数组转换为PIL图像
#         img = Image.fromarray(img_array_normalized)
        
#         # 构建输出文件的路径
#         output_file = os.path.join(output_folder, filename + '.png')
        
#         # 保存为PNG格式
#         img.save(output_file)

# # 设置DICOM文件的文件夹路径和输出文件夹路径
# dicom_folder_path = r'C:\Users\allstar\Desktop\aaaa\DICOM (2)\DICOM\ST0\SE5'
# output_folder_path = r'C:\Users\allstar\Desktop\aaaa\ct_png'


# # 调用函数进行转换
# convert_dicom_to_png(dicom_folder_path, output_folder_path)

import os
import pydicom
from PIL import Image
import numpy as np
import cv2


# windowing 函数
def windowing(img, window_width, window_center):
    minwindow = float(window_center) - 0.5 * float(window_width)
    new_img = (img - minwindow) / float(window_width)
    
     # 将归一化后的图像缩放到0到255
    new_img = new_img * 255
    
    # 确保结果在0到255之间
    new_img = np.clip(new_img, 0, 255)
    
    # 转换为整数类型
    new_img = new_img.astype(np.uint8)
    return new_img

def save_dcm_as_png(dcm_folder, output_folder):
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
        img_array = ds.pixel_array  # 获取图像数据
        
        img_array[img_array>2429] = 2429
        
        # 获取DICOM图像的窗宽和窗位，用于归一化
        
        # max_value = np.max(img_array)
        # min_value = np.min(img_array)
    
        # window_level = (max_value+min_value)/2
        # window_width = abs(max_value-window_level)
    
    
        # 归一化到0-255范围
        # img_array_normalized = np.clip((img_array - window_level) / (window_width / 255), 0, 255)
        
        img_array_normalized = windowing(img_array, 400, 30)
                 
        contours, _ = cv2.findContours(img_array_normalized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 初始化最大外接矩形的变量
        max_area = 0
        max_rect = None

        # 遍历所有轮廓
        for contour in contours:
            # 计算轮廓的外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            # 计算矩形的面积
            area = w * h
            # 更新最大外接矩形
            if area > max_area:
                max_area = area
                max_rect = (x, y, w, h)

        # 绘制最大外接矩形
        if max_rect:
            x, y, w, h = max_rect
            img_array_normalized = img_array_normalized[y:y+h, x:x+w]
        


        # 将图像数据转换为 PIL 图像
        img = Image.fromarray(img_array_normalized)
        
    

        # 保存为 PNG 文件
        output_path = os.path.join(output_folder, f"ct_{i+1}.png")
        img.save(output_path)
        print(f"保存 {output_path}")

# 使用示例
dcm_folder = r'D:\TJ_DATA_CT\DICOM\PA0\ST0\SE3'  # DICOM 文件夹路径
output_folder = r'D:\TJ_data_T\ct_png'  # 输出文件夹路径

save_dcm_as_png(dcm_folder, output_folder)

