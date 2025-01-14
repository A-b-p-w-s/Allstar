"""
没用的

"""

import time
import random

for i in range(0,10000000):
    random_number = random.uniform(0.5, 2)
    time.sleep(random_number)
    
    print(f"epoch:{i}/{10000000} [-----iou:{round(i/10000000+(random_number)/1000,4)}-----] time:{round((i+1)*random_number,4)} s")
  


# ========================================================================================================
    
import open3d as o3d
import numpy as np

def adaptive_alpha(point_cloud, k=10):
    """
    计算自适应Alpha值。
    
    参数:
    point_cloud -- Open3D的PointCloud对象。
    k -- 用于计算每个点的k近邻的数目。
    
    返回:
    adaptive_alpha_value -- 计算得到的自适应Alpha值。
    """
    # 将PointCloud转换为numpy数组
    points = np.asarray(point_cloud.points)
    
    # 计算k近邻的距离
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    distances, indices = kdtree.search_knn_vector_3d(points, k)
    
    # 计算每个点的k近邻的平均距离
    average_distances = np.mean(distances, axis=1)
    
    # 计算全局平均距离作为Alpha值
    adaptive_alpha_value = np.mean(average_distances)
    
    return adaptive_alpha_value

# 加载点云数据
point_cloud = o3d.io.read_point_cloud("path_to_your_point_cloud.ply")

# 计算自适应Alpha值
alpha_value = adaptive_alpha(point_cloud, k=10)
print(f"自适应Alpha值: {alpha_value}")

# 使用计算得到的Alpha值创建Alpha Shape
alpha_shape = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha_value)

# 可视化Alpha Shape
o3d.visualization.draw_geometries([alpha_shape])

# ========================================================================================================



IF(AND(NOT(NOT([当前时间]>[结束时间])),IFERROR(FIND([状态],"进行中"),FALSE())),"超时！！！","正常")  #超时判断

IF(IF(AND(NOT(NOT([当前时间]>[结束时间])),IFERROR(FIND([状态],"进行中"),FALSE())),TRUE(),FALSE()),0,MIN(MAX(IF(IFERROR(FIND([状态],"进行中"),FALSE()),(([当前时间]-[开始时间]+1) / ([结束时间]-[开始时间]+1)),IF(IFERROR(FIND([状态],"已完成"),FALSE()),1,0)),0),1))  #进度

