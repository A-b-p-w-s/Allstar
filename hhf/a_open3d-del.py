"""
测试可视化三维点数据进行凹包优化后可视化的代码
"""
import open3d as o3d
import numpy as np
import glob
import pydicom
import os

def read_dicom_series(directory):
    # 获取文件夹中所有.dcm文件的路径
    # dcm_files = sorted(glob.glob(os.path.join(directory, '*.dcm'))) 
    dcm_files = sorted(glob.glob(os.path.join(directory,'*'))) 
    # 读取第一个DICOM文件以获取图像尺寸和深度信息
    ref_image = pydicom.dcmread(dcm_files[0])

    pixel_spacing = ref_image.PixelSpacing # 返回一个包含x和y距离的元组

    # 获取体素的z方向距离（层间距）
    slice_thickness = ref_image.SliceThickness # 返回z方向的距离
    pixdim = [None] * 4
    pixdim[1]=pixel_spacing[0]
    pixdim[2]=pixel_spacing[1]
    pixdim[3]=slice_thickness
    
    
    # 初始化一个3D数组来存储所有DICOM图像
    images = np.stack([pydicom.dcmread(f).pixel_array for f in dcm_files])
    
    return np.transpose(images, (2, 1, 0)),pixdim




points_a,pixdim=read_dicom_series(r'C:\Users\allstar\Desktop\abc\213-new\liver-213')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def find_concave_hull(points, alpha):
    """
    通过 Alpha Shape 算法计算凹包
    """
    if len(points) < 4:
        return points
    
    tri = Delaunay(points)
    triangles = points[tri.simplices]
    edges = set()
    for triangle in triangles:
        for i in range(3):
            edge = tuple(sorted([tuple(triangle[i]), tuple(triangle[(i + 1) % 3])]))
            edges.add(edge)

    # 基于 alpha 距离移除较长边
    edges = [edge for edge in edges if np.linalg.norm(np.array(edge[0]) - np.array(edge[1])) < alpha]
    hull_points = np.array(list(set([point for edge in edges for point in edge])))

    return hull_points
for i in range(points_a.shape[2]):
    # 读取图像并预处理
    image = np.array(points_a[:,:,i]*255)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = np.vstack(contours[i][:, 0] for i in range(len(contours)))

    # 计算凹包
    alpha = 20  # 调节此值影响凹包的松紧程度

    concave_hull = find_concave_hull(points, alpha)

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap='gray')
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=1, label='Points')
    plt.plot(np.append(concave_hull[:, 0], concave_hull[0, 0]), 
            np.append(concave_hull[:, 1], concave_hull[0, 1]), 
            color='red', linewidth=2, label='Concave Hull')
    plt.legend()
    plt.title('Concave Hull Visualization')
    plt.show()



points = np.where(points > 0)

# 使用 NumPy 的向量化操作来创建障碍点的数组
points = np.vstack((points,)).T
print(points.shape)

# 创建点云对象
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# 可视化点云
o3d.visualization.draw_geometries([point_cloud])