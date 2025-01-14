"""
测试计算穿刺路径中路劲与肝脏三角形平面的角度
"""
# import time
# import random

# for i in range(0,10000000):
#     random_number = random.uniform(0.5, 2)
#     time.sleep(random_number)
    
#     print(f"epoch:{i}/{10000000} [-----iou:{round(i/10000000+(random_number)/1000,4)}-----] time:{round((i+1)*random_number,4)} s")


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ray_intersects_triangleb(ray_origin, ray_direction, triangle_vertices):
    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction)
    
    ray_direction = ray_direction - ray_origin
    ray_direction = ray_direction / np.linalg.norm(ray_direction)  # 单位化方向
    
    vertex0, vertex1, vertex2 = np.array(triangle_vertices[0]), np.array(triangle_vertices[1]), np.array(triangle_vertices[2])

    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0

    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if -1e-6 < a < 1e-6:
        return False, None  # 射线与三角形平行，无交点

    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False, None  # 交点在三角形外部

    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if v < 0.0 or u + v > 1.0:
        return False, None  # 交点在三角形外部

    t = f * np.dot(edge2, q)
    if t > 1e-6:  # 交点在射线上
        intersection_point = ray_origin + t * ray_direction
        return True, intersection_point
    else:
        return False, None  # 交点在射线起点之前
    

# def plot_ray_and_triangle(ray_origin, ray_direction, triangle_vertices, intersects, intersection_point=None):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # 绘制三角形
#     ax.plot([triangle_vertices[0][0], triangle_vertices[1][0]],
#             [triangle_vertices[0][1], triangle_vertices[1][1]],
#             [triangle_vertices[0][2], triangle_vertices[1][2]], 'g-')
#     ax.plot([triangle_vertices[1][0], triangle_vertices[2][0]],
#             [triangle_vertices[1][1], triangle_vertices[2][1]],
#             [triangle_vertices[1][2], triangle_vertices[2][2]], 'g-') 
#     ax.plot([triangle_vertices[2][0], triangle_vertices[0][0]],
#             [triangle_vertices[2][1], triangle_vertices[0][1]],
#             [triangle_vertices[2][2], triangle_vertices[0][2]], 'g-')

#     # 绘制射线
#     ax.quiver(ray_origin[0], ray_origin[1], ray_origin[2], ray_direction[0], ray_direction[1], ray_direction[2], color='b')

#     # 标记交点
#     if intersects:
#         ax.scatter(*intersection_point, color='r')
#         ax.text(intersection_point[0], intersection_point[1], intersection_point[2], 'Intersection Point', color='r')

#     plt.show()

def plot_ray_and_triangle(ray_origin, ray_direction, ray_list,aaa, intersects, intersection_point=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for triangle_vertices in ray_list:
        # 绘制三角形
        ax.plot([triangle_vertices[0][0], triangle_vertices[1][0]],
                [triangle_vertices[0][1], triangle_vertices[1][1]],
                [triangle_vertices[0][2], triangle_vertices[1][2]], 'g-')
        ax.plot([triangle_vertices[1][0], triangle_vertices[2][0]],
                [triangle_vertices[1][1], triangle_vertices[2][1]],
                [triangle_vertices[1][2], triangle_vertices[2][2]], 'g-')
        ax.plot([triangle_vertices[2][0], triangle_vertices[0][0]],
                [triangle_vertices[2][1], triangle_vertices[0][1]],
                [triangle_vertices[2][2], triangle_vertices[0][2]], 'g-')

    # ax.plot([aaa[0][0], aaa[1][0]],
    #         [aaa[0][1], aaa[1][1]],
    #         [aaa[0][2], aaa[1][2]], 'b-')
    # ax.plot([aaa[1][0], aaa[2][0]],
    #         [aaa[1][1], aaa[2][1]],
    #         [aaa[1][2], aaa[2][2]], 'b-')
    # ax.plot([aaa[2][0], aaa[0][0]],
    #         [aaa[2][1], aaa[0][1]],
    #         [aaa[2][2], aaa[0][2]], 'b-')

    # 绘制射线
    ax.quiver(ray_origin[0], ray_origin[1], ray_origin[2], ray_direction[0], ray_direction[1], ray_direction[2], color='b')
    
    ax.scatter(511*pixdim[1],511*pixdim[2],209, color='g')
    ax.scatter(0,0,0, color='b')
    
    
    ax.scatter(272*pixdim[2],272*pixdim[1],98*pixdim[3], color='y')
    ax.scatter(1*pixdim[2], 1*pixdim[1], 1*pixdim[3], color='y')
    
    
    
   
    #代码计算穿刺路径 红色
    ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2], color='r')
    
    # ax.plot([272*pixdim[1],intersection_point[0]], [272*pixdim[2],intersection_point[1]], [98*pixdim[3],intersection_point[2]], color='r')
    
    #真实计算穿刺路径 蓝色
    ax.plot([1*pixdim[2],272*pixdim[2]], [1*pixdim[1],272*pixdim[1]], [1*pixdim[3],98*pixdim[3]], color='b')
    
    
    # ax.quiver(1*pixdim[2], 1*pixdim[1], 1*pixdim[3], 272*pixdim[2],272*pixdim[1],98*pixdim[3], color='r')

    # 标记交点
    if intersects:
        pass
        # ax.quiver(272*pixdim[2], 272*pixdim[1], 98*pixdim[3], intersection_point[0], intersection_point[1], intersection_point[2], color='r')
        
        ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2], color='r')
        # ax.text(intersection_point[0], intersection_point[1], intersection_point[2], 'Intersection Point', color='r')

    plt.show()


import cv2
import numpy as np
import nibabel as nib
import math
import open3d as o3d


def ray_intersects_triangle(ray_origin, ray_direction, v0,v1,v2):
    
    # 将输入转换为NumPy数组以便于计算
    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction)
    ray_direction = ray_direction - ray_origin
    ray_direction = ray_direction / np.linalg.norm(ray_direction)  # 单位化方向
    
    vertex0, vertex1, vertex2 = np.array(v0), np.array(v1), np.array(v2)

    # 计算三角形的边向量
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0

    # 计算射线方向与三角形边的叉积
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if -1e-6 < a < 1e-6:
        return None  # 射线与三角形平行，无交点

    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return None  # 交点在三角形外部

    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if v < 0.0 or u + v > 1.0:
        return None  # 交点在三角形外部

    # 计算交点的t值
    t = f * np.dot(edge2, q)
    if t > 1e-6:  # 交点在射线上
        intersection_point = ray_origin + t * ray_direction
        return intersection_point.tolist()
    else:
        return None  # 交点在射线起点之前

# 加载NIfTI图像
nii_image3 = nib.load(r'C:\Users\allstar\Desktop\a_new_model_body\model_body\body.nii.gz')  # 替换为你的文件路径

# nii_image3 = nib.load(r'C:\Users\allstar\Desktop\test\p\0.nii.gz')
body_data = nii_image3.get_fdata()

pixdim = nii_image3.header['pixdim']
approx_list=[]

for i in range(body_data.shape[2]):
    # 提取特定切片并转换为灰度图
    gray = (body_data[:, :, i] * 255).astype(np.uint8)

    # 二值化
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制原始轮廓和近似轮廓
    for c in contours:
        # 近似轮廓
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        for a in approx:
            new_approx = np.concatenate((a[0]*pixdim[1], [i*pixdim[3]]))
            approx_list.append(new_approx)


import open3d as o3d
import numpy as np

# 将numpy数组转换为open3d的PointCloud对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(approx_list)

ray_list = []
# points = []

# 使用alpha shapes算法生成表面
alpha = 400  # alpha值越小，生成的表面越精细
try:
    alpha_shape_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    alpha_shape_mesh.compute_vertex_normals()
    
     # 定义两点连线的端点
    # point1 = np.array([ 262*pixdim[2], 142*pixdim[1], 59*pixdim[3] ]).reshape(1, 3)
    # point2 = np.array([ 58*pixdim[2], 384*pixdim[1], 54*pixdim[3] ]).reshape(1, 3)
    
    # point1 = np.array([ 272*pixdim[2], 234*pixdim[1], 98*pixdim[3] ]).reshape(1, 3)
    # point2 = np.array([ 196*pixdim[2], 53*pixdim[1], 60*pixdim[3] ]).reshape(1, 3)
    point1 = np.array([ 272*pixdim[2],272*pixdim[1],98*pixdim[3]]).reshape(1, 3)
    point2 = np.array([ 1*pixdim[2], 1*pixdim[1], 1*pixdim[3] ]).reshape(1, 3)


    # 检查线段与每个三角形面的交点
    intersected_faces = []
    

    for i, triangle in enumerate(alpha_shape_mesh.triangles):
        # 获取三角形的顶点
        v0 = alpha_shape_mesh.vertices[triangle[0]]
        v1 = alpha_shape_mesh.vertices[triangle[1]]
        v2 = alpha_shape_mesh.vertices[triangle[2]]
        # points.append(v0)
        # points.append(v1)
        # points.append(v2)
        ray_list.append([v0,v1,v2])
                
        # 计算线段与三角形的交点
        # intersect_point = ray_intersects_triangle([ 142*pixdim[1], 262*pixdim[2],59*pixdim[3]], [ 138*pixdim[1],384*pixdim[2],54*pixdim[3]], v0, v1, v2)
        
        intersect_point = ray_intersects_triangle(point1[0], point2[0], v0, v1, v2)
        

        # _,intersect_point = ray_intersects_triangleaaa(np.array(point1[0]), np.array(point2[0]), np.array(v0), np.array(v1), np.array(v2))
        
        if intersect_point is not None:
            
            aaa = [v0,v1,v2]
            
            axis_point01 = o3d.geometry.PointCloud()
            axis_point01.points = o3d.utility.Vector3dVector(v0.reshape(1, 3))
            axis_point02 = o3d.geometry.PointCloud()
            axis_point02.points = o3d.utility.Vector3dVector(v1.reshape(1, 3))
            axis_point03 = o3d.geometry.PointCloud()
            axis_point03.points = o3d.utility.Vector3dVector(v2.reshape(1, 3))
            axis_point04 = o3d.geometry.PointCloud()
            axis_point04.points = o3d.utility.Vector3dVector(np.array(intersect_point).reshape(1, 3))
            
            # 创建两个表示坐标轴点的点云
            axis_point1 = o3d.geometry.PointCloud()
            axis_point1.points = o3d.utility.Vector3dVector(point1)

            axis_point2 = o3d.geometry.PointCloud()
            axis_point2.points = o3d.utility.Vector3dVector(point2)
                        
            # 创建一个线段连接两个点
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector(point1.tolist() + point2.tolist())
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            
            intersected_faces.append((i, intersect_point))
            

    # 如果有交点，计算线段与面的夹角
    if intersected_faces:
        face_index, intersect_point = intersected_faces[0]
        # print(len(intersected_faces))
        
        
        triangle = [alpha_shape_mesh.vertices[idx] for idx in alpha_shape_mesh.triangles[face_index]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        

        # # 计算线段的方向向量
        # line_direction = point2 - point1
        
        # 计算线段的方向向量
        line_direction = (point2 - point1) / np.linalg.norm(point2 - point1)
        
        # 计算射线方向向量与平面法向量的夹角的余弦值
        cos_angle = np.dot(line_direction, normal) / (np.linalg.norm(line_direction) * np.linalg.norm(normal))

        # 计算射线方向向量与平面法向量的夹角（弧度）
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # 射线与平面的角度是射线方向向量与平面法向量夹角的补角
        angle_with_plane_rad = np.pi / 2 - angle_rad
        

        # 将弧度转换为度
        angle_with_plane_deg = np.degrees(angle_with_plane_rad)

        # # 计算线段与面的夹角
        # angle = np.arccos(np.dot(line_direction, normal) / (np.linalg.norm(line_direction) * np.linalg.norm(normal)))
        
        # angle_deg = np.degrees(angle)

        print(f"Line intersects face {face_index} at point {intersect_point}")
        
        print(f"The angle between the line and the face is {angle_with_plane_deg} degrees.")    
        
        print(f'distance is {np.linalg.norm(point1[0]-np.array(intersect_point))}mm')
    
    # 提取每个面的顶点坐标并保存到列表中
    faces = []
    for triangle in alpha_shape_mesh.triangles:
        face = [
            alpha_shape_mesh.vertices[triangle[0]],
            alpha_shape_mesh.vertices[triangle[1]],
            alpha_shape_mesh.vertices[triangle[2]]
        ]
        faces.append(face)    
    
    # 可视化生成的表面、两个坐标轴点和连接它们的线
    print("Visualizing the reconstructed surface with the two axis points and the line...")
    o3d.visualization.draw_geometries([ alpha_shape_mesh,axis_point1, axis_point2,line,axis_point01,axis_point02,axis_point03,axis_point04], window_name="Reconstructed Surface with Axis Points and Line", width=800, height=600)
    
    
except Exception as e:
    print("Error creating alpha shape:", e)


A = point1[0]
B = point2[0]

import numpy as np

def calculate_plane_normal(vertex0, vertex1, vertex2):
    # 计算三角形的两个边向量
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    # 计算平面的法向量
    normal = np.cross(edge1, edge2)
    return normal

def vector_angle(vec1, vec2):
    # 计算两个向量之间的夹角
    dot_product = np.dot(vec1, vec2)
    magnitude_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    # 夹角的余弦值
    cos_angle = dot_product / magnitude_product
    # 夹角的弧度值
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return angle_rad

def ray_triangle_plane_angle(ray_origin, ray_direction, triangle_vertices):
    # 计算三角形平面的法向量
    normal = calculate_plane_normal(*triangle_vertices)
    # 确保射线方向向量是从A指向B
    ray_direction = np.array(ray_direction)
    if np.allclose(ray_origin, (0, 0, 0)):  # 假设ray_origin是原点
        ray_direction = ray_direction / np.linalg.norm(ray_direction)  # 标准化方向向量
    else:
        ray_direction = (ray_direction - ray_origin) / np.linalg.norm(ray_direction - ray_origin)  # 标准化方向向量
    # 计算射线与平面法向量的夹角
    angle_rad = vector_angle(ray_direction, normal)
    # 射线与平面的角度是射线与法向量的夹角的补角
    angle_with_plane = np.pi / 2 - angle_rad
    # 将弧度转换为度
    angle_deg = np.degrees(angle_with_plane)
    return angle_deg


angle = ray_triangle_plane_angle(A, B, aaa)
print(f"射线与三角形平面的角度是 {angle:.2f} 度")

intersects, point = ray_intersects_triangleb(A, B, aaa)
plot_ray_and_triangle(A, B, ray_list,aaa, intersects, point if intersects else None)







# from scipy.spatial import distance
# dists = distance.pdist(points, 'euclidean')
# # 将距离转换成方阵形式
# dist_matrix = distance.squareform(dists)
# # 找到最大距离
# print(np.max(dist_matrix))
# print(np.mean(dist_matrix))
# print(np.min(dist_matrix))
# plot_ray_and_triangle(A, B, aaa, intersects, point if intersects else None)

