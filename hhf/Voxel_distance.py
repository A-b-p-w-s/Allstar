"""
计算体像距离代码
用于计算CT图像中各个器官的真实体积

"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2



# 1. 读取.nii.gz文件
image_path = r'C:\Users\allstar\Desktop\body_model\boundary_new\boundary.nii.gz'
image_obj = nib.load(image_path)
image_data = image_obj.get_fdata()

x = image_obj.header['pixdim']
print(f'X轴体素尺寸: {x[1]} mm')
print(f'Y轴体素尺寸: {x[2]} mm')
print(f'Z轴体素尺寸: {x[3]} mm')


# 获取图像的维度信息
z_dim, x_dim, y_dim = image_data.shape

num_pixdim_size=0
for slice_index in range(y_dim):
    # 获取特定z轴的切片
    slice_data = image_data[:, :, slice_index]
    pixdim_size=np.sum(slice_data>0)*x[1]*x[2]*x[3]
    # print(pixdim_size)
    num_pixdim_size += pixdim_size
    # print(f"{pixdim_size}cm3")
    slice_data = np.flipud(np.transpose(slice_data))
    slice_data = np.uint8(slice_data)
    # 处理或显示切片数据
    # 例如，这里我们打印每个切片的最小值和最大值
    min_val = np.min(slice_data)
    max_val = np.max(slice_data)
    
    if max_val==1:

        _, slice_data = cv2.threshold(slice_data, 0.5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(slice_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area=0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # 如果找到了轮廓，计算最大面积的外接矩形
        if max_contour is not None:
            # 获取最小外接矩形
            rect = cv2.minAreaRect(max_contour)

            # 获取矩形的四个顶点
            box = cv2.boxPoints(rect)
            box = np.int_(box)  # 转换为整数型
            # 绘制最大面积的外接矩形
            cv2.drawContours(slice_data, [box], 0, (150, 150, 150), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(slice_data, f'Volume of liver section:{round(pixdim_size,2)}mm3', (5, 50), font, 0.7, (150, 150, 150), 1)
        
            # 可视化处理后的切片
            plt.imshow(slice_data)
            plt.colorbar()
            plt.show()
print(f'{num_pixdim_size}mm3')

