"""
测试同济联影截图数据对齐到dcm数据的代码

"""
import os
import pydicom
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2



def convert_dicom_to_png(dicom_path, output_folder,masek_folder_path):
    # 确保输出文件夹存在
    if not os.path.exists(dicom_path):
        os.makedirs(dicom_path)

    i=1
    for filename1 in os.listdir(masek_folder_path):
        png_file = cv2.imread(os.path.join(masek_folder_path,filename1))
        edges = cv2.Canny(png_file, threshold1=0, threshold2=100)
        end_y,end_x = edges.shape[:2]
        edges = edges[30:end_y-35, end_x-1060:end_x-125]
        png_file = png_file[30:end_y-35, end_x-1060:end_x-125]
        
    
                
        ct_file = cv2.imread(os.path.join(output_folder,f'ct_{i}.png'))
        CT_h,CT_w = ct_file.shape[:2]
        
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
        
        gray_mask = (png_file[:,:,0] == png_file[:,:,1]) & (png_file[:,:,1] == png_file[:,:,2])

        # # 将灰度部分的像素设置为黑色，非灰度部分保持不变
        png_file[gray_mask] = [0, 0, 0]
        png_file = cv2.resize(png_file, (CT_w, CT_h), interpolation=cv2.INTER_AREA)

        
        l_w = 0
        r_w = 512-CT_w
        t_h = 0
        b_h = 512-CT_h
        
        # print(f'W:{l_w},{r_w},H:{t_h},{b_h}')
        

        png_file = cv2.copyMakeBorder(png_file, t_h, b_h, l_w, r_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # gray_mask = (png_file[:,:,0] == png_file[:,:,1]) & (png_file[:,:,1] == png_file[:,:,2])

        # # 将灰度部分的像素设置为黑色，非灰度部分保持不变
        # png_file[gray_mask] = [0, 0, 0]
        
        # print(png_file.shape)
        
        # cv2.imshow('png_file',png_file)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # 将numpy数组转换为PIL图像
        img = Image.fromarray(png_file.astype(np.uint8))
        
        # 构建输出文件的路径
        output_file = os.path.join(dicom_path, f'ct_{i}' + '.png')
        
        # 保存为PNG格式
        img.save(output_file)
       
        i+=1
        
        

# 设置DICOM文件的文件夹路径和输出文件夹路径
dicom_folder_path = r'D:\TJ_data_T\dicom_png' ##savepng
output_folder_path = r'D:\TJ_data_T\ct_png' ##CTpng

masek_folder_path = r'D:\TJ_DATA_JT\DICOM_PA0\part_1' ##masekpng

# 调用函数进行转换
convert_dicom_to_png(dicom_folder_path, output_folder_path,masek_folder_path)