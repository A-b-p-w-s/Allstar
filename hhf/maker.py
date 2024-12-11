# import nibabel as nib
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # 打开.nii.gz文件
# nii_file = r'C:\Users\allstar\Desktop\bbb\ct_001_1.3.12.2.1107.5.1.4.73332.30000024081423272882300087618_0000.nii.gz'
# img = nib.load(nii_file)

# # 获取数据
# data = img.get_fdata()

# # 检查数据维度
# print("数据维度:", data.shape)

# # 假设数据是3D的，我们有一个轴是切片轴
# slice_axis = 2  # 这取决于你的数据，可能需要调整


# # windowing 函数
# def windowing(img, window_width, window_center):
#     minwindow = float(window_center) - 0.5 * float(window_width)
#     new_img = (img - minwindow) / float(window_width)
#     return new_img

# # 显示每个切片
# for i in range(data.shape[slice_axis]): 
#     # 读取图像并转换为灰度图
#     image = data[:, :, i]
#     image = windowing(image, 50, 0)
#     # image = cv2.GaussianBlur(image.astype('uint8'), (5, 5), 0)
#     image = cv2.adaptiveThreshold(image.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#     # image[image < 0] = 0
#     # image[image > 1] = 1
#     # image = (image * 255).astype('uint8')
#     # print(i)



#     # 应用高斯模糊去除噪声
#     image = cv2.GaussianBlur(image.astype('uint8'), (5, 5), 0)
#     # Canny边缘检测
#     edges = cv2.Canny(image, 0, 75)

#     # 霍夫圆检测
#     circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 5, param1=50, param2=25, minRadius=0, maxRadius=20)

#     # 绘制圆形和圆心
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:
#             cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
#             cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
#             print(i[0], i[1])

#     # 显示结果
#     cv2.imshow('Detected Circles', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    

# --------------------------------------------- maker ---------------------------------------------


import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 打开.nii.gz文件
mask_nii_file = r'C:\Users\allstar\Desktop\bbb\Segmentation.nii'
mask_nii_file2 = r'C:\Users\allstar\Desktop\bbb\mask4.nii'
nii_file = r'C:\Users\allstar\Desktop\bbb\ct_001_1.nii.gz'
img = nib.load(nii_file)

image_data = img.get_fdata()
pixdim = img.header['pixdim']
print('-------------------------------------------------------')
print(pixdim[1],pixdim[2],pixdim[3])
mask_img = nib.load(mask_nii_file)
mask_img2 = nib.load(mask_nii_file2)
# 获取数据
data = img.get_fdata()
mask = mask_img.get_fdata()
mask2 = mask_img2.get_fdata()

# mask = mask-1
# mask[mask<0]=1

# data = data*data

# -----------------------------------------
# data = np.where(mask == 1, -1024, data)

# data = np.abs(data)

# 检查数据维度
print("数据维度:", data.shape)

# 假设数据是3D的，我们有一个轴是切片轴
slice_axis = 2  # 这取决于你的数据，可能需要调整



# windowing 函数
def windowing(img, window_width, window_center):
    minwindow = float(window_center) - 0.5 * float(window_width)
    new_img = (img - minwindow) / float(window_width)
    return new_img

processed_slices = []

X=[]

# 显示每个切片
for i in range(data.shape[slice_axis]): 
    # 读取图像并转换为灰度图
    image = data[:, :,i]
    mask_data = mask[:, :,i]
    mask_data2 = mask2[:,:,i]
    
    z=i
    
    mask_data = cv2.dilate(mask_data, np.ones((3, 3), np.uint8), iterations=5)
    mask_data2 = cv2.dilate(mask_data2, np.ones((3, 3), np.uint8), iterations=5)
    
    image = np.where(mask_data == 1, -1024, image)
    image = np.where(mask_data2 == 1, -1024, image)
    
    image = windowing(image,500,150)
 
    
    # image = np.where(mask_data, image, 0)
    
    # image[image<0]=0
    
    
    image = cv2.adaptiveThreshold(image.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 3)
    
    
    processed_slices.append(255 - image)
    
    
    edges = cv2.Canny(image.astype('uint8'), 0, 75)

    # 霍夫圆检测
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=15, minRadius=8, maxRadius=20)

    # 绘制圆形和圆心
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # print('-----------------------------------')
        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (128, 128, 128), 2)
            cv2.circle(image, (i[0], i[1]), 2, (128, 128, 128), 3)
            # print(i[0], i[1],z)
            
            #添加坐标
            X.append([i[1]*pixdim[1],(512-i[0])*pixdim[2],(210-z)*pixdim[3]])
            # X.append([i[1],i[0],z])
            
    
    
    # # 显示结果
    # cv2.imshow('Detected Circles', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# 选择聚类数，这里我们根据数据生成的中心数来选择
k = 4

# 创建 KMeans 实例
kmeans = KMeans(n_clusters=k)

# 拟合模型
kmeans.fit(X)

# 预测数据集的聚类标签
捕获的标签 = kmeans.predict(X)

# 绘制聚类结果
# plt.scatter(X[:, 0], X[:, 1], c=捕获的标签, s=50, cmap='viridis')

# 绘制聚类中心
centers = (kmeans.cluster_centers_)

# 使用 ceil 函数向上取整
rounded_data = np.ceil(centers)

# 将结果转换为整数类型
integer_data = rounded_data.astype(int)

print(centers)



# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)

# plt.show()
    
    
# # 将处理后的切片堆叠回3D数组
# processed_data = np.stack(processed_slices, axis=slice_axis)

# print(np.unique(processed_data))

# # 确保数据类型与原始数据类型相同
# processed_data = processed_data.astype(data.dtype)

# # 创建一个新的Nifti1Image对象
# new_img = nib.Nifti1Image(processed_data, img.affine, img.header)

# # 保存处理后的图像
# output_file = r'C:\Users\allstar\Desktop\bbb\processed_image_abs.nii.gz'
# nib.save(new_img, output_file)
    