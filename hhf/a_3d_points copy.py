import numpy as np
import cv2
import nibabel as nib
import open3d as o3d


def ray_intersects_triangle(ray_origin, ray_direction, v0,v1,v2):
    
    # 将输入转换为NumPy数组以便于计算
    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction)
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

nii_image3 = nib.load(r'C:\Users\allstar\Desktop\test\p\0.nii.gz')
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

# 使用alpha shapes算法生成表面
alpha = 50  # alpha值越小，生成的表面越精细
try:
    alpha_shape_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    alpha_shape_mesh.compute_vertex_normals()
    
     # 定义两点连线的端点
    point1 = np.array([ 262*pixdim[2], 142*pixdim[1], 59*pixdim[3] ]).reshape(1, 3)
    point2 = np.array([ 384*pixdim[2], 58*pixdim[1], 94*pixdim[3] ]).reshape(1, 3)


    # 检查线段与每个三角形面的交点
    intersected_faces = []
    

    for i, triangle in enumerate(alpha_shape_mesh.triangles):
        # 获取三角形的顶点
        v0 = alpha_shape_mesh.vertices[triangle[0]]
        v1 = alpha_shape_mesh.vertices[triangle[1]]
        v2 = alpha_shape_mesh.vertices[triangle[2]]

        
        # 计算线段与三角形的交点
        intersect_point = ray_intersects_triangle(point1[0], point2[0], v0, v1, v2)
        if intersect_point is not None:      
            intersected_faces.append((i, intersect_point))
            

    # 如果有交点，计算线段与面的夹角
    if intersected_faces:
        face_index, intersect_point = intersected_faces[0]
        triangle = [alpha_shape_mesh.vertices[idx] for idx in alpha_shape_mesh.triangles[face_index]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])

        # 计算线段的方向向量
        line_direction = point2 - point1
        
        # 计算射线方向向量与平面法向量的夹角的余弦值
        cos_angle = np.dot(line_direction, normal) / (np.linalg.norm(line_direction) * np.linalg.norm(normal))

        # 计算射线方向向量与平面法向量的夹角（弧度）
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # 射线与平面的角度是射线方向向量与平面法向量夹角的补角
        angle_with_plane_rad = np.pi / 2 - angle_rad

        # 将弧度转换为度
        angle_with_plane_deg = np.degrees(angle_with_plane_rad)

        print(f"Line intersects face {face_index} at point {intersect_point}")
        
        print(f"The angle between the line and the face is {angle_with_plane_deg} degrees.")    
    
except Exception as e:
    print("Error creating alpha shape:", e)