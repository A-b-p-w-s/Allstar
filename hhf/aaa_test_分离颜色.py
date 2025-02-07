"""
同济联影截图mask对齐后提取个器官颜色标签的代码
用于将图mask对齐后提取个器官颜色标签生成图片
"""
import cv2
import numpy as np
import os

if __name__ == '__main__':
    
    folder_path = r'D:\TJ_data_T\ct_png'
    output_folder = r'D:\TJ_data_T\fg'
    
    # 遍历文件夹
    for i in range(len(os.listdir(folder_path))):
        # 检查文件名是否以.png结尾
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, f'ct_{i+1}.png')
        # 读取图像·
        img = cv2.imread(file_path) 
        # 将图像转换为HSV颜色空间
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        #肝脏
        lower = np.array([80, 25, 25])  # 浅蓝色的低阈值 
        upper = np.array([120, 255, 255])  # 浅蓝色的高阈值
        
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
        black_image = np.zeros_like(img)

        # 将棕色区域提取到黑色背景上
        brown_region = cv2.bitwise_and(img, img, mask=mask)
        

        output_path = os.path.join(output_folder, f"ct_{i+1}.png")
        # 保存结果
        cv2.imwrite(output_path, brown_region)
    print('====ok=====')

        # # 显示结果
        # cv2.imshow("Brown Region", brown_region)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()