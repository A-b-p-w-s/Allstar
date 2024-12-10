import os
import pydicom
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2



def convert_dicom_to_png(dicom_path, output_folder,masek_folder_path):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    i=1
    for filename1 in os.listdir(masek_folder_path):
        png_file = cv2.imread(os.path.join(masek_folder_path,filename1))
        edges = cv2.Canny(png_file, threshold1=0, threshold2=100)
        end_y,end_x = edges.shape[:2]
        edges = edges[30:end_y-35, end_x-1060:end_x-125]
        png_file = png_file[30:end_y-35, end_x-1060:end_x-125]
        
        
        ct_file = cv2.imread(os.path.join(output_folder,f'IM_{i}.png'))
        ct_edges = cv2.Canny(ct_file, threshold1=0, threshold2=100)
        end_y,end_x = ct_edges.shape[:2]
        ct_edges = ct_edges[0:end_y-60, 0:end_x]
        
        
        # # ========================================================================计算大小========================================================================
        # 找到所有值为255的像素点
        white_pixels = np.where(ct_edges == 255)

        # 初始化变量来存储第一次和最后一次出现255的位置
        first_vertical = first_horizontal = last_vertical = last_horizontal = None

        # 从左到右（水平方向）
        if white_pixels[1].size > 0:
            first_horizontal = min(white_pixels[1])  # 第一次出现255的位置
            last_horizontal = max(white_pixels[1])  # 最后一次出现255的位置

        # 从上到下（垂直方向）
        if white_pixels[0].size > 0:
            first_vertical = min(white_pixels[0])    # 第一次出现255的位置
            last_vertical = max(white_pixels[0])    # 最后一次出现255的位置
            
        CT_w=last_horizontal-first_horizontal
        CT_h=last_vertical-first_vertical
        print(f"CT_W：{CT_w} CT_H：{CT_h}")
       # ========================================================================计算大小========================================================================
        
        ct_edges = ct_edges[first_vertical:last_vertical, first_horizontal:last_horizontal]
        
        
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
        print(f"PNG_W：{png_w} PNG_H：{png_h}")
        # ========================================================================计算大小========================================================================
        
        # edges = edges[first_vertical2:last_vertical2, first_horizontal2:last_horizontal2] 
        # edges = cv2.resize(edges, (CT_w, CT_h), interpolation=cv2.INTER_AREA)
        
        png_file = png_file[first_vertical2:last_vertical2, first_horizontal2:last_horizontal2]
        png_file = cv2.resize(png_file, (CT_w, CT_h), interpolation=cv2.INTER_AREA)
        
        
        edges[edges > 0]=255
        
        
        l_w = first_horizontal
        r_w = 512-last_horizontal
        t_h = first_vertical
        b_h = 512-last_vertical
        
        print(f'W:{l_w},{r_w},H:{t_h},{b_h}')
        

        png_file = cv2.copyMakeBorder(png_file, t_h, b_h, l_w, r_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        gray_mask = (png_file[:,:,0] == png_file[:,:,1]) & (png_file[:,:,1] == png_file[:,:,2])

        # 将灰度部分的像素设置为黑色，非灰度部分保持不变
        png_file[gray_mask] = [0, 0, 0]
        
        print(png_file.shape)
        
        cv2.imshow('png_file',png_file)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

       
        i+=1
        
        

# 设置DICOM文件的文件夹路径和输出文件夹路径
dicom_folder_path = r'C:\Users\allstar\Desktop\aaaa\dicom_png'
output_folder_path = r'C:\Users\allstar\Desktop\aaaa\ct_png'

masek_folder_path = r'C:\Users\allstar\Desktop\aaaa\DICOM (2)\DICOM\jietu'

# 调用函数进行转换
convert_dicom_to_png(dicom_folder_path, output_folder_path,masek_folder_path)