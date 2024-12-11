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
        edges = cv2.Canny(png_file, threshold1=0, threshold2=200)
        end_y,end_x = edges.shape[:2]
        edges = edges[30:end_y-35, end_x-1060:end_x-125]
        png_file = png_file[30:end_y-35, end_x-1060:end_x-125]
        
        # ========================================================================计算大小========================================================================
        # # 找到所有值为255的像素点
        # white_pixels = np.where(edges == 255)

        # # 初始化变量来存储第一次和最后一次出现255的位置
        # first_vertical = first_horizontal = last_vertical = last_horizontal = None

        # # 从左到右（水平方向）
        # if white_pixels[1].size > 0:
        #     first_horizontal0 = min(white_pixels[1])  # 第一次出现255的位置
        #     last_horizontal0 = max(white_pixels[1])  # 最后一次出现255的位置

        # # 从上到下（垂直方向）
        # if white_pixels[0].size > 0:
        #     first_vertical0 = min(white_pixels[0])    # 第一次出现255的位置
        #     last_vertical0 = max(white_pixels[0])    # 最后一次出现255的位置
        # PNG_w=last_horizontal0-first_horizontal0
        # PNG_h=last_vertical0-first_vertical0
            
        # print(f"PNG_W：{PNG_w} PNG_H：{PNG_h}")
        # ========================================================================计算大小========================================================================
        
        
        ct_file = cv2.imread(os.path.join(output_folder,f'IM_{i}.png'))
        ct_edges = cv2.Canny(ct_file, threshold1=0, threshold2=200)
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
       
        # print(CT_w/PNG_w,CT_h/PNG_h) 
        
        height, width = edges.shape[:2]

        # 计算新的尺寸，缩小1.76倍
        new_width = int(width / 1.76)
        new_height = int(height / 1.76)

        # 等比例缩小图片
        resized_image = cv2.resize(edges, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        png_file = cv2.resize(png_file, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # print(np.unique(resized_image))
        
        resized_image[resized_image > 0]=255
        # ========================================================================计算大小========================================================================
         # 找到所有值为255的像素点
        white_pixels = np.where(resized_image == 255)

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
        
        resized_image = resized_image[first_vertical2:last_vertical2, first_horizontal2:last_horizontal2] 
        png_file = png_file[first_vertical2:last_vertical2, first_horizontal2:last_horizontal2]
        
        print(f'W:{511-png_w},H:{511-png_h}')
        
        
        
        l_w = first_horizontal
        r_w = 511-last_horizontal
        t_h = first_vertical
        b_h = 511-last_vertical
        
        print(f'W:{l_w},{r_w},H:{t_h},{b_h}')
        
        W_L = int((l_w/(l_w+r_w))*(512-png_w))
        W_R = int((r_w/(l_w+r_w))*(512-png_w))
        
        H_T = int((t_h/(t_h+b_h))*(512-png_h))
        H_B = int((b_h/(t_h+b_h))*(512-png_h))

        
        print(f'W-l:{W_L},W-r:{W_R}')
        print(f'H-T:{H_T},H-B:{H_B}')
        # print(resized_image.shape[1]+W_L+W_R)
        # print(resized_image.shape[0]+H_T+H_B)
        if resized_image.shape[1]+W_L+W_R<512:
            W_R=W_R+(512-(resized_image.shape[1]+W_L+W_R))
        if resized_image.shape[0]+H_T+H_B<512:
            H_B=H_B+(512-(resized_image.shape[0]+H_T+H_B))

        png_file = cv2.copyMakeBorder(png_file, H_T, H_B, W_L, W_R, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        print(png_file.shape)
        # cv2.imshow('png_file', png_file)
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
dicom_folder_path = r'C:\Users\allstar\Desktop\aaaa\dicom_png'
output_folder_path = r'C:\Users\allstar\Desktop\aaaa\ct_png'

masek_folder_path = r'C:\Users\allstar\Desktop\aaaa\DICOM (2)\DICOM\jietu'

# 调用函数进行转换
convert_dicom_to_png(dicom_folder_path, output_folder_path,masek_folder_path)