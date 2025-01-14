"""
测试完整路劲规划的copy代码
"""

#============================================old_nii============================================
#============================================old_nii============================================

import numpy as np
import nibabel as nib
from tqdm import tqdm
import open3d as o3d
import cv2
import os
import pydicom
import nibabel as nib
import glob
        

def read_dicom_series(directory):
    # 获取文件夹中所有.dcm文件的路径
    # dcm_files = sorted(glob.glob(os.path.join(directory, '*.dcm'))) 
    dcm_files = sorted(glob.glob(os.path.join(directory, '*'))) 
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
    
    return np.transpose(images, (1, 2, 0)),pixdim

def optimization_points(nii_data):
    array= []
    for i in range(nii_data.shape[2]):
        # 提取特定切片并转换为灰度图
        gray = (nii_data[:, :, i] * 255).astype(np.uint8)
        
        
        edges = cv2.Canny(gray, 100, 200)/255

    
        array.append(edges.astype(int))
 
    array2 = []   
    for i in range(nii_data.shape[0]):
        # 提取特定切片并转换为灰度图
        gray = (nii_data[i, :, :] * 255).astype(np.uint8)
        
        
        edges = cv2.Canny(gray, 100, 200)/255
        
        array2.append(edges.astype(int))
        
    array2 = np.array(array2)
    array = np.array(array)

    array = array.transpose((1, 2, 0))
    
    array = array + array2

    array[array>0] = 1
    return array
    

planning_points=[]

def planning_3D( radius = 1 , p_len = 3 , source = np.array([293 ,285 ,112]) , targets = np.array([[377  ,412 ,77 ],[147  ,432 ,120  ]]) , bone_path = r"E:\Data\Nii\213\bone-213\CT_001_1.2.3.4.5_0000.nii.gz" , vessel_path = r"E:\Data\Nii\213\vessel-213\CT_006_1.2.3.4.5_0000.nii.gz" , liver_path = r"E:\Data\Nii\213\liver-213\CT_003_1.2.3.4.5_0000.nii.gz"):   

    import  numpy as np

    # 找到x、y、z轴上的最小值和最大值
    min_x, min_y, min_z = targets.min(axis=0)
    max_x, max_y, max_z = targets.max(axis=0)
    x_values = np.arange(min_x, max_x, 10)
    y_values = np.arange(min_y, max_y, 10)
    z_values = np.arange(min_z, max_z, 1)

    # 将网格展平为一维数组
    points_in_box = np.array(np.meshgrid(x_values, y_values, z_values, indexing='ij')).T.reshape(-1, 3)

    targets = points_in_box


    ##===================================================bone===================================================
    
    
    nii_data2,pixdim =read_dicom_series(bone_path)


    nii_data2 = optimization_points(nii_data2)

    # 获取所有像素值为1的点的索引
    indices2 = np.where(nii_data2 > 0)

    # 使用 NumPy 的向量化操作来创建障碍点的数组
    points2 = np.vstack((indices2,)).T


    ##===================================================vessel===================================================
    # nii_img = load_dicom_series(r'E:\Data\Dicom\213-new\vessel-213')  # 替换为你的文件路径
    # nii_img = nib.load(vessel_path)
    
    # nii_data = nii_img.get_fdata()
    nii_data,_ = read_dicom_series(vessel_path)

    nii_data = optimization_points(nii_data)

    # 获取所有像素值为1的点的索引
    indices = np.where(nii_data > 0)

    # 使用 NumPy 的向量化操作来创建障碍点的数组
    points = np.vstack((indices,)).T
    obstacles = np.concatenate((points, points2), axis=0)
    

    # unique_obstacles, indices, inverse, counts = np.unique(obstacles, axis=0, return_index=True, return_inverse=True, return_counts=True)

    # pixdim = nii_img.header['pixdim']

    # 计算射线与障碍球的交点
    def ray_intersects(obstacles, source, directions):
        # 计算射线与球的交点
        oc = obstacles - source
        a = np.sum(directions**2, axis=1)
        b = 2 * np.sum(directions * oc, axis=1)
        c = np.sum(oc**2, axis=1) - radius**2
        discriminant = b**2 - 4*a*c
        
        # 计算t1和t2，只考虑非负的解
        t1 = (-b[discriminant>=0] - np.sqrt(discriminant[discriminant>=0])) / (2*a)
        t2 = (-b[discriminant>=0] + np.sqrt(discriminant[discriminant>=0])) / (2*a)
        
        # 选择有效的t值
        t = np.maximum(t1, t2)
        
        # 检查t是否大于0
        valid_t = t > 0
        
        return valid_t

    # 计算每条射线碰到的障碍球数量
    intersection_counts = []
    for target in tqdm(targets):
        direction = (source - target) / np.linalg.norm(source - target)  # 单位方向向量
        intersects = ray_intersects(obstacles, source, direction[None, :])  # 扩展方向向量
        count = np.sum(intersects)
        intersection_counts.append(count)
        
        
    # 输出每条射线碰到的障碍球数量
    # print(intersection_counts)
    zero_indices  = np.where( np.array(intersection_counts) == 0) #没有碰到障碍的下标
    distances = np.linalg.norm(targets - source, axis=1) # 计算每个路径的距离
    

    #满足H1、H2条件的点集
    new_distances = distances[zero_indices]
    H2_dict  = dict(zip(zero_indices[0],new_distances))
    H1H2_dict  = {key: value for key, value in H2_dict.items() if value < 200}
    H1H2_keys = [item[0] for item in sorted(H1H2_dict.items(), key=lambda item: item[1])]
    # print(targets[H1H2_keys])


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


    ##===================================================liver===================================================

    nii_image3 = nib.load(r'C:\Users\allstar\Desktop\abc\213\liver-213\CT_003_1.2.3.4.5_0000.nii.gz')
    body_data = nii_image3.get_fdata()
    # print(body_data,"======================================================")
    
    body_data,_ = read_dicom_series(liver_path)
    # print(body_data)
    # pixdim = nii_image3.header['pixdim']

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


    # 使用alpha shapes算法生成表面
    alpha = 400  # alpha值越小，生成的表面越精细


    
    try:
        alpha_shape_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        alpha_shape_mesh.compute_vertex_normals()
        
        point1 = np.array([ source[1]*pixdim[2], source[0]*pixdim[1], source[2]*pixdim[3] ]).reshape(1, 3)
        
        
        
        for point in targets[H1H2_keys]:
            
            #=============================================================================================
            
            # 1 0 2
            # 定义两点连线的端点
            point2 = np.array([ point[1]*pixdim[2], point[0]*pixdim[1], point[2]*pixdim[3] ]).reshape(1, 3)


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
                    
                    aaa = [v0,v1,v2]
                    intersected_faces.append((i, intersect_point))
                                    


            # 如果有交点，计算线段与面的夹角
            if intersected_faces:
                face_index, intersect_point = intersected_faces[0]
                
                triangle = [alpha_shape_mesh.vertices[idx] for idx in alpha_shape_mesh.triangles[face_index]]
                normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                
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

                # print(f"Line intersects face {face_index} at point {intersect_point}")
                
                # print(f"The angle between the line and the face is {angle_with_plane_deg} degrees.")    
                
                # print(f'distance is {np.linalg.norm(point1[0]-np.array(intersect_point))}mm')

                if angle_with_plane_deg>20:
                    # print(point2)
                    # print(f'distance is {np.linalg.norm(point1[0]-np.array(intersect_point))}mm')
                    p_len -=1 
                    planning_points.append([int(point2[0][1]/pixdim[2]),int(511-point2[0][0]/pixdim[1]),int(173-point2[0][2]/pixdim[3])])
                    if p_len <= 0:
                        return planning_points
                # point2 #穿刺点坐标
                
                # o3d.visualization.draw_geometries([ alpha_shape_mesh,axis_point1, axis_point2,line,axis_point01,axis_point02,axis_point03,axis_point04], window_name="Reconstructed Surface with Axis Points and Line", width=800, height=600)
        return []
    except Exception as e:
        pass
        # print("Error creating alpha shape:", e)
    
    

if __name__ == "__main__":
    
    radius = 5
    p_len = 3
    source =  np.array([293 ,285 ,112])
    targets = np.array([[377  ,412 ,77 ],[147  ,432 ,120  ]])
    # bone_path = r"C:\Users\allstar\Desktop\abc\213\bone-213\CT_001_1.2.3.4.5_0000.nii.gz"
    # vessel_path = r"C:\Users\allstar\Desktop\abc\213\vessel-213\CT_006_1.2.3.4.5_0000.nii.gz"
    # liver_path = r"C:\Users\allstar\Desktop\abc\213\ct-213\CT_002_1.2.3.4.5_0000.nii.gz"
    
    
    bone_path = r"C:\Users\allstar\Desktop\233\d_m-213"
    vessel_path = r"C:\Users\allstar\Desktop\233\g_j-213"
    liver_path = r"C:\Users\allstar\Desktop\233\liver-213"
     
    planning_points = planning_3D( radius = radius , p_len = p_len , source = source , targets = targets , bone_path = bone_path , vessel_path = vessel_path , liver_path = liver_path)
    print(planning_points)
    
# [[287, 99, 66], [277, 99, 66], [277, 99, 56]]