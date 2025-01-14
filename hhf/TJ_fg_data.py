"""
同济医院截图转换成dcm代码
用于将同济医院联影系统的截图与原本dcm数据对齐后转换成dcm

"""

import os
import pydicom
from PIL import Image
import numpy as np
import cv2


# windowing 函数
def windowing(img, window_width, window_center):
    
    # 获取原始范围
    orig_min = np.min(img)
    orig_max = np.max(img)
    target_max = img.max()
    target_min = -2048
  
    # 映射到目标范围
    if img.min() >= 0:
        img = ((img - orig_min) / (orig_max - orig_min)) * (target_max - target_min) + target_min
    
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    # print(img.min(),img.max())
    img = np.clip(img, min_val, max_val)
    # factor = 255.0 / (max_val - min_val)
    # img = ((img - min_val) * factor).astype(np.uint8)
    # img_clipped  = np.clip(img, 0, 255)
    img_mapped = ((img - min_val) / (max_val - min_val))*255

    # 转换为 uint8 数据类型
    new_img = img_mapped.astype(np.uint8)
    
    
    # minwindow = float(window_center) - 0.5 * float(window_width)
    # new_img = (img - minwindow) / float(window_width)
    
    #  # 将归一化后的图像缩放到0到255
    # new_img = new_img * 255
    
    # # 确保结果在0到255之间
    # new_img = np.clip(new_img, 0, 255)
    
    # # 转换为整数类型
    # new_img = new_img.astype(np.uint8)
    return new_img

def save_dcm_as_png(dcm_folder, masek_folder_path,output_folder,lower,upper):
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
    i = 0
    # 遍历排序后的 DICOM 文件并保存为 PNG
    for (ds, _),filename1 in zip(dicom_data,os.listdir(masek_folder_path)):
        #==============================================================================================分割截图图片
        png_file = cv2.imread(os.path.join(masek_folder_path,filename1))
        edges = cv2.Canny(png_file, threshold1=0, threshold2=100)
        end_y,end_x = edges.shape[:2]
        edges = edges[30:end_y-35, end_x-1060:end_x-125]
        png_file = png_file[30:end_y-35, end_x-1060:end_x-125]
        
        
        
        # ========================================================================计算大小========================================================================
         # 找到所有值为255的像素点
        white_pixels = np.where(edges == 255)

        # 初始化变量来存储第一次和最后一次出现255的位置
        first_vertical2 = first_horizontal2 = last_vertical2 = last_horizontal2 = None

        # 从左到右（水平方向）
        if white_pixels[1].size > 0:
            first_horizontal2 = min(white_pixels[1])  # 第一次出现255的位置
            last_horizontal2 = max(white_pixels[1])  # 最后一次出现255的位置

        # 从上到下（垂直方向）
        if white_pixels[0].size > 0:
            first_vertical2 = min(white_pixels[0])    # 第一次出现255的位置
            last_vertical2 = max(white_pixels[0])    # 最后一次出现255的位置
            
        png_w=last_horizontal2-first_horizontal2
        png_h=last_vertical2-first_vertical2
        # ========================================================================计算大小========================================================================

        
        png_file = png_file[first_vertical2:last_vertical2, first_horizontal2:last_horizontal2]
        
        # cv2.imshow('png_file', png_file)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        gray_mask = (png_file[:,:,0] == png_file[:,:,1]) & (png_file[:,:,1] == png_file[:,:,2])

        # # 将灰度部分的像素设置为黑色，非灰度部分保持不变
        png_file[gray_mask] = [0, 0, 0]
        
        #=============================================================================================== dcm原始数据
        img_array = ds.pixel_array  # 获取图像数据
        
        # img_array[img_array>2429] = 2429
        
        img_array_normalized = windowing(img_array,800, 40)
        
                 
        contours, _ = cv2.findContours(img_array_normalized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 初始化最大外接矩形的变量
        max_area = 0
        max_rect = None
        CT_w = CT_h = 0
        t_h = b_h = l_w = r_w =0
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
            CT_w = w
            CT_h = h
            l_w = x
            r_w = 512-x-w
            t_h = y
            b_h = 512-y-h
            img_array_normalized = img_array_normalized[y:y+h, x:x+w]
            
            # cv2.imshow('img_array_normalized', img_array_normalized)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        
        png_file = cv2.resize(png_file, (CT_w, CT_h), interpolation=cv2.INTER_AREA)
        png_file = cv2.copyMakeBorder(png_file, t_h, b_h, l_w, r_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # 将图像数据转换为 PIL 图像
        img = Image.fromarray(png_file)
        
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)        

        #肝脏
        # lower = np.array([80, 25, 25])  # 浅蓝色的低阈值 
        # upper = np.array([120, 255, 255])  # 浅蓝色的高阈值
        
        #动脉
        # lower = np.array([120, 50, 50])  # 紫色的低阈值
        # upper = np.array([160, 255, 255])  # 紫色的高阈值
        
        #门静脉
        # lower = np.array([20, 60, 60])  # 黄色的低阈值
        # upper = np.array([40, 255, 255])  # 黄色的高阈值
        
        #肝静脉
        # lower = np.array([5, 50, 200])   # 浅橙色的低阈值
        # upper = np.array([15, 255, 255])  # 浅橙色的高阈值
        
        #下腔脉
        # lower = np.array([10, 100, 20])   # 棕色的低阈值
        # upper = np.array([25, 255, 200])  # 棕色的高阈值
        
        #病灶
        # lower = np.array([80, 50, 50])   # 蓝绿色的低阈值
        # upper = np.array([160, 255, 255])  # 蓝绿色的高阈值

        # 创建掩码，提取棕色区域
        mask = cv2.inRange(hsv_image, lower, upper)

        # 创建一个全黑的图像
        mask_image = np.zeros_like(img)

        # 将棕色区域提取到黑色背景上
        mask_image = cv2.bitwise_and(img, img, mask=mask)
        
        
        png_file = cv2.cvtColor(mask_image, cv2.COLOR_HSV2BGR)
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
        output_dcm_path = os.path.join(output_folder, f"IM{i}.dcm")
        new_ds.save_as(output_dcm_path)
        i = i+1
        print(f"保存新的 DICOM 文件：{output_dcm_path}")

# 使用示例

dcm_folder = r'D:\TJ_DATA_CT\DICOMBD\PA0\ST0\SE3'       # DICOM 文件夹路径
masek_folder_path = r'D:\TJ_DATA_JT\DICOMBD_PA0\part_'  # masek_png
output_folder = r'D:\TJ_data_T\DICOMBD\PA0'             # 输出文件夹路径


lower = [np.array([80, 25, 25]),np.array([120, 50, 50]),np.array([20, 60, 60]),np.array([5, 50, 200]),np.array([10, 100, 20]),np.array([80, 50, 50])]
upper = [np.array([120, 255, 255]),np.array([160, 255, 255]),np.array([40, 255, 255]),np.array([15, 255, 255]),np.array([25, 255, 200]),np.array([160, 255, 255])]
# 肝脏 动脉 门静脉 静脉 下腔脉 瘤变
save_name = ['liver','artery','portal_vein','hepatic_vein','inferior_vena_cava','tumor']
for i in range(6): 
    masek_folder_path_a = masek_folder_path+str(i+1)
    output_folder_a = os.path.join(output_folder, save_name[i])
    save_dcm_as_png(dcm_folder, masek_folder_path_a,output_folder_a,lower[i],upper[i])
print("转换完成！")