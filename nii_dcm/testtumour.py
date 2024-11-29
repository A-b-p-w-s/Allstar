import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# def fit_circle(points):
#     # 将点的坐标转换为NumPy数组
#     points = np.asarray(points)
    
#     # 计算点的均值
#     centroid = points.mean(axis=0)
    
#     # 初始化U矩阵
#     U = np.zeros((2, 2))
    
#     # 遍历所有点，计算U矩阵
#     for x, y in points:
#         U += (x - centroid[0]) * np.array([[x, y], [y, -x]])
    
#     # 计算U矩阵的特征值和特征向量
#     eigenvalues, eigenvectors = np.linalg.eig(U)
    
#     # 选择与最大特征值对应的特征向量作为外接圆的半径和圆心偏移
#     radius = np.sqrt(max(eigenvalues))
#     offset = eigenvectors[:, np.argmax(eigenvalues)].real
    
#     # 计算圆心坐标
#     center = centroid - offset * radius
    
#     return center, radius

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# 打开.nii.gz文件
nii_file = r'C:\Users\allstar\Desktop\body_model\new_tumour\tumour.nii.gz'
img = nib.load(nii_file)

data = img.get_fdata()

slice_axis = 2

X=[]

for i in range(data.shape[slice_axis]): 
    # 读取图像并转换为灰度图
    image = data[:, :,i]
    
    z = i
    
    image = image*255
    # 拟合外接圆
    # center, radius = fit_circle(points)
    
    
    # 应用高斯模糊，减少图像噪声
    # image = cv2.GaussianBlur(image, (9, 9), 2)
    
    image = cv2.adaptiveThreshold(image.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 3)
    edges = cv2.Canny(image, 0, 75)

    # 使用霍夫圆变换检测圆
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                                param1=50, param2=15, minRadius=10, maxRadius=20)

    # 将圆心和半径转换为整数
    # circles = circles.astype(int)

    # 遍历检测到的圆
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            
            # 绘制圆心
            cv2.circle(image, (i[0], i[1]), 2, (128, 128, 128), -1)
            # 绘制圆轮廓
            # cv2.circle(image, (i[0], i[1]), i[2], (128, 128, 128), 3)
            
            #添加坐标
            X.append([i[1],i[0],z])
        
    # # 显示结果
    # cv2.imshow('Detected Circles', image)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift



# 创建 MeanShift 对象
mean_shift = MeanShift(bandwidth=10, bin_seeding=True)

# 训练模型
mean_shift.fit(X)

# 获取聚类中心
cluster_centers = mean_shift.cluster_centers_

k = len(cluster_centers)
# print(k)

# # 选择聚类数，这里我们根据数据生成的中心数来选择
# k = 16

# 创建 KMeans 实例
kmeans = KMeans(n_clusters=k)

# 拟合模型
kmeans.fit(X)

# 预测数据集的聚类标签
捕获的标签 = kmeans.predict(X)

# 绘制聚类中心
centers = (kmeans.cluster_centers_)

# 使用 ceil 函数向上取整
rounded_data = np.ceil(centers)

# 将结果转换为整数类型
integer_data = rounded_data.astype(int)

print(integer_data)

