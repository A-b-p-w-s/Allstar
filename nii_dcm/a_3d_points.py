import cv2
import numpy as np
import nibabel as nib
import math
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


def get_line_triangle_intersection(p1, p2, v0, v1, v2):
    """
    Calculate the intersection point of a line segment and a triangle.
    Parameters:
    p1: np.array, the starting point of the line segment.
    p2: np.array, the ending point of the line segment.
    v0: np.array, a vertex of the triangle.
    v1: np.array, another vertex of the triangle.
    v2: np.array, the third vertex of the triangle.
    Returns:
    intersect_point: np.array or None, the intersection point if it exists.
    """
    # Convert inputs to numpy arrays if they aren't already
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    v0 = np.asarray(v0)
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # Find vectors along two edges of the triangle
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Compute the direction vector of the line segment
    direction = p2 - p1

    # Compute the cross product of direction vector and edge2
    cross_h = np.cross(direction, edge2)

    # Compute the dot product of cross_h and edge1
    a = np.dot(cross_h, edge1)

    # If a is close to zero, the line is parallel to the triangle or edge1
    if np.isclose(a, 0, atol=1e-6):
        return None

    # Compute the inverse of a
    inv_a = 1.0 / a

    # Compute the vector from p1 to v0
    start_vec = v0 - p1

    # Compute the u parameter of the intersection point
    u = inv_a * np.dot(cross_h, start_vec)

    # If u is outside the range [0, 1], the intersection is outside the triangle
    if u < 0 or u > 1:
        return None

    # Compute the q vector
    q = np.cross(start_vec, edge1)

    # Compute the v parameter of the intersection point
    v = inv_a * np.dot(q, direction)

    # If v is outside the range [0, 1], the intersection is outside the triangle
    if v < 0 or u + v > 1:
        return None

    # Compute the t parameter of the intersection point
    t = inv_a * np.dot(q, edge2)

    # If t is outside the range [0, 1], the intersection is outside the line segment
    if t < 0 or t > 1:
        return None

    # The intersection point is inside the triangle and on the line segment
    intersect_point = p1 + t * direction

    return intersect_point



# 加载NIfTI图像
# nii_image3 = nib.load(r'C:\Users\allstar\Desktop\body_model\body_model\Segmentation.nii')  # 替换为你的文件路径

source = np.array([271 ,186 ,153])

nii_image3 = nib.load(r'C:\Users\allstar\Desktop\test\p\0.nii.gz')
body_data = nii_image3.get_fdata()

pixdim = nii_image3.header['pixdim']
approx_list=[]

for i in range(body_data.shape[2]):
    # 提取特定切片并转换为灰度图
    gray = (body_data[:, :, i] * 255).astype(np.uint8)

    # source_tumour = np.array([186 ,271])
    source_tumour = np.array([250 ,90])

    targets = np.array([400,130])

    # 二值化
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制原始轮廓和近似轮廓
    for c in contours:
        # 近似轮廓
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        # cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)  # 绘制近似轮廓
        for a in approx:
            new_approx = np.concatenate((a[0]*pixdim[1], [i*pixdim[3]]))
            approx_list.append(new_approx)


# ====================================================================================================================================================================

# import open3d as o3d
# import numpy as np


# # 将numpy数组转换为open3d的PointCloud对象
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.array(approx_list))

# # 使用alpha shapes算法生成表面
# alpha = 50  # alpha值越小，生成的表面越精细
# try:
#     alpha_shape_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
#     alpha_shape_mesh.compute_vertex_normals()

#     # 可视化生成的表面
#     print("Visualizing the reconstructed surface...")
#     o3d.visualization.draw_geometries([alpha_shape_mesh], window_name="Reconstructed Surface", width=800, height=600)

#     # 如果需要保存生成的3D模型，可以使用以下代码
#     o3d.io.write_triangle_mesh("output_mesh.ply", alpha_shape_mesh)
# except Exception as e:
#     print("Error creating alpha shape:", e)


# print(len(approx_list))


# ====================================================================================================================================================================


import open3d as o3d
import numpy as np

# 将numpy数组转换为open3d的PointCloud对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(approx_list)

# 使用alpha shapes算法生成表面
alpha = 50  # alpha值越小，生成的表面越精细
try:
    alpha_shape_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    alpha_shape_mesh.compute_vertex_normals()
    
    # # 创建两个表示坐标轴点的点云
    # axis_point1 = o3d.geometry.PointCloud()
    # a1=np.array([ 262*pixdim[2],142*pixdim[1] ,59*pixdim[3]]).reshape(1, 3)
    # a2=np.array([384*pixdim[2], 138*pixdim[1],54*pixdim[3] ]).reshape(1, 3)
    # axis_point1.points = o3d.utility.Vector3dVector(a1)

    # axis_point2 = o3d.geometry.PointCloud()
    # axis_point2.points = o3d.utility.Vector3dVector(a2)

    # # 创建一个线段连接两个点
    # line = o3d.geometry.LineSet()
    # line.points = o3d.utility.Vector3dVector(a1.tolist() + a2.tolist())
    # line.lines = o3d.utility.Vector2iVector([[0, 1]])

    # # 可视化生成的表面、两个坐标轴点和连接它们的线
    # print("Visualizing the reconstructed surface with the two axis points and the line...")
    # o3d.visualization.draw_geometries([alpha_shape_mesh, axis_point1, axis_point2, line], window_name="Reconstructed Surface with Axis Points and Line", width=800, height=600)
    
    
     # 定义两点连线的端点
    point1 = np.array([ 262*pixdim[2],142*pixdim[1] ,59*pixdim[3]]).reshape(1, 3)
    point2 = np.array([384*pixdim[2], 138*pixdim[1],54*pixdim[3] ]).reshape(1, 3)
    


    # 创建一个线段连接两个点
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(point1.tolist() + point2.tolist())


    # 检查线段与每个三角形面的交点
    intersected_faces = []
    

    for i, triangle in enumerate(alpha_shape_mesh.triangles):
        # 获取三角形的顶点
        v0 = alpha_shape_mesh.vertices[triangle[0]]
        v1 = alpha_shape_mesh.vertices[triangle[1]]
        v2 = alpha_shape_mesh.vertices[triangle[2]]
        # 计算线段与三角形的交点
        # intersect_point = get_line_triangle_intersection(point1[0], point2[0], v0, v1, v2)
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

        # 计算线段与面的夹角
        angle = np.arccos(np.dot(line_direction, normal) / (np.linalg.norm(line_direction) * np.linalg.norm(normal)))
        angle_deg = np.degrees(angle)

        print(f"Line intersects face {face_index} at point {intersect_point}")
        print(f"The angle between the line and the face is {angle_deg,2} degrees.")



    # 创建两个表示坐标轴点的点云
    axis_point1 = o3d.geometry.PointCloud()
    a1=np.array([ 262*pixdim[2],142*pixdim[1] ,59*pixdim[3]]).reshape(1, 3)
    a2=np.array([384*pixdim[2], 138*pixdim[1],54*pixdim[3] ]).reshape(1, 3)
    axis_point1.points = o3d.utility.Vector3dVector(a1)

    axis_point2 = o3d.geometry.PointCloud()
    axis_point2.points = o3d.utility.Vector3dVector(a2)
    
    
    # 提取每个面的顶点坐标并保存到列表中
    faces = []
    for triangle in alpha_shape_mesh.triangles:
        face = [
            alpha_shape_mesh.vertices[triangle[0]],
            alpha_shape_mesh.vertices[triangle[1]],
            alpha_shape_mesh.vertices[triangle[2]]
        ]
        faces.append(face)

    # # 输出每个面的顶点坐标
    # print("\nFaces of the reconstructed surface with their vertex coordinates:")
    # for i, face in enumerate(faces):
    #     print(f"Face {i}: {face}")

    
    axis_point01 = o3d.geometry.PointCloud()
    axis_point01.points = o3d.utility.Vector3dVector(faces[797][0].reshape(1, 3))
    axis_point02 = o3d.geometry.PointCloud()
    axis_point02.points = o3d.utility.Vector3dVector(faces[797][1].reshape(1, 3))
    axis_point03 = o3d.geometry.PointCloud()
    axis_point03.points = o3d.utility.Vector3dVector(faces[797][2].reshape(1, 3))
    

    # 创建一个线段连接两个点
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(a1.tolist() + a2.tolist())
    line.lines = o3d.utility.Vector2iVector([[0, 1]])

    # 可视化生成的表面、两个坐标轴点和连接它们的线
    print("Visualizing the reconstructed surface with the two axis points and the line...")
    o3d.visualization.draw_geometries([alpha_shape_mesh, axis_point1, axis_point2, line,axis_point01,axis_point02,axis_point03], window_name="Reconstructed Surface with Axis Points and Line", width=800, height=600)
    
    

    
except Exception as e:
    print("Error creating alpha shape:", e)