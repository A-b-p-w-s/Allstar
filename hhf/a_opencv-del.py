"""
测试可视化三维点云的代码
"""
# import nibabel as nib
# import numpy as np
# from scipy.spatial import ConvexHull
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # nii_image = nib.load(r'C:\Users\allstar\Desktop\body_model\body_model\Segmentation.nii')  # 替换为你的文件路径
# nii_image = nib.load(r'C:\Users\allstar\Desktop\test\p\9.nii.gz')  # 替换为你的文件路径
# body_data = nii_image.get_fdata()

# array2d=[]
# array2d_z=[]
# for i in range(body_data.shape[2]):
#     if np.sum(body_data[:, :, i]):
#         array2d.append(body_data[:, :, i])
#         array2d_z.append(i)



# array2d_one=[]
# for i in range(len(array2d)):
#     indices = np.where(np.array(array2d[i])>0)

#     coordinates = [(x, y) for x, y in zip(indices[0], indices[1])]
#     [array2d_one.append(coordinates[a]+ (array2d_z[i],)) for a in ConvexHull(coordinates).vertices]

# x = [point[0] for point in array2d_one]
# y = [point[1] for point in array2d_one]
# z = [point[2] for point in array2d_one]

# # 创建一个新的matplotlib图和一个3D子图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 在3D空间内绘制散点图
# ax.scatter(x,y,z,alpha=0.5)


# # 设置轴标签
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# # 显示图形
# plt.show()

# ====================================================================================================================================================================

# import nibabel as nib

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial import ConvexHull

# # 假设 hull_3d 是你已经计算好的凸包，这里我们随机生成一些点来模拟


# nii_image = nib.load(r'C:\Users\allstar\Desktop\test\p\9.nii.gz')  # 替换为你的文件路径
# # nii_image = nib.load(r'C:\Users\allstar\Desktop\body_model\body_model\Segmentation.nii')  # 替换为你的文件路径
# body_data = nii_image.get_fdata()

# indices = np.where(body_data >0)
# points = np.array([[x, y, z] for x, y, z in zip(indices[0], indices[1], indices[2])])

# # points = np.random.rand(30, 3)  # 30个随机的三维点
# # print(points)

# # 计算凸包
# hull = ConvexHull(points)

# # 提取构成凸包的顶点
# vertices = hull.vertices

# # 提取构成凸包的三角形
# simplices = hull.simplices

# # 创建一个3D图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制构成凸包的点
# # ax.scatter(points[:,0], points[:,1], points[:,2])

# # 绘制构成凸包的三角形
# for simplex in simplices:
#     ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')


# aaa=points[vertices]

# # 绘制凸包的顶点
# ax.scatter(aaa[:,0], aaa[:,1], aaa[:,2], color='r')

# # 设置图形的标签
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')


# # 在3D空间内绘制散点图
# # ax.scatter(x,y,z,alpha=0.5)


# # 显示图形
# plt.show()

# ====================================================================================================================================================================

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial import ConvexHull, Delaunay

# # 假设points是一个包含三维坐标的数组
# points = np.random.rand(100, 3)

# # 计算凸包
# hull = ConvexHull(points)

# # 计算Delaunay三角剖分
# tri = Delaunay(points)

# # 创建图形和三维子图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制点
# ax.scatter(points[:,0], points[:,1], points[:,2])

# # 绘制凸包
# for simplex in hull.simplices:
#     ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')

# # 绘制三角剖分
# for simplex in tri.simplices:
#     ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'b-')

# # 设置标签和显示
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()


# ====================================================================================================================================================================
import numpy as np
import pyvista as pv

import nibabel as nib


# nii_image = nib.load(r'C:\Users\allstar\Desktop\test\p\0.nii.gz')  # 替换为你的文件路径
# # nii_image = nib.load(r'C:\Users\allstar\Desktop\body_model\body_model\Segmentation.nii')  # 替换为你的文件路径
# body_data = nii_image.get_fdata()

# indices = np.where(body_data >0)
# points = np.array([[x, y, z] for x, y, z in zip(indices[0], indices[1], indices[2])])

# points = points.astype(np.float32)

# points = np.random.rand(100, 3)  # 随机生成100个点


# # 创建点云对象
# point_cloud = pv.PolyData(points)

# # 从点云重建表面
# mesh = point_cloud.reconstruct_surface()


# # 绘制点云
# point_cloud.plot(eye_dome_lighting=True, show_scalar_bar=False)

# # 保存网格为STL文件
# mesh.save('mesh.stl')

# # 绘制生成的网格
# mesh.plot(color='orange')

# ====================================================================================================================================================================


import numpy as np
import open3d as o3d

# 生成随机点云数据
# points = np.random.rand(10000, 3)  # 生成10000个随机点
import nibabel as nib

nii_image = nib.load(r'C:\Users\allstar\Desktop\aaaaa\liver-213\ct_213_0000.nii.gz')  # 替换为你的文件路径
# nii_image = nib.load(r'C:\Users\allstar\Desktop\body_model\body_model\Segmentation.nii')  # 替换为你的文件路径
body_data = nii_image.get_fdata()

pixdim = nii_image.header['pixdim']

indices = np.where(body_data >0)
points = np.array([[x*pixdim[1], y*pixdim[2], z*pixdim[3]] for x, y, z in zip(indices[0], indices[1], indices[2])])

points = points.astype(np.float32)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)


# 可视化点云数据
o3d.visualization.draw_geometries([point_cloud])


# 估计点云的法线
point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 使用Poisson方法进行表面重建
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)

# 可视化重建的表面
o3d.visualization.draw_geometries([mesh])


# ====================================================================================================================================================================

# 论文截图用
# import numpy as np
# import open3d as o3d
# import nibabel as nib
# import cv2


# def optimization_points(nii_data):
#     array= []
#     for i in range(nii_data.shape[2]):
#         # 提取特定切片并转换为灰度图
#         gray = (nii_data[:, :, i] * 255).astype(np.uint8)
        
        
#         edges = cv2.Canny(gray, 100, 200)/255

    
#         array.append(edges.astype(int))
 
#     array2 = []   
#     for i in range(nii_data.shape[0]):
#         # 提取特定切片并转换为灰度图
#         gray = (nii_data[i, :, :] * 255).astype(np.uint8)
        
        
#         edges = cv2.Canny(gray, 100, 200)/255
        
#         array2.append(edges.astype(int))
        
#     array2 = np.array(array2)
#     array = np.array(array)

#     array = array.transpose((1, 2, 0))
    
#     array = array + array2

#     array[array>0] = 1
#     return array

# # 加载 NIfTI 图像文件
# nii_image = nib.load(r'C:\Users\allstar\Desktop\aaaaa\liver-213\ct_213_0000.nii.gz')  # 替换为你的文件路径
# body_data = nii_image.get_fdata()

# # 获取体素尺寸
# pixdim = nii_image.header['pixdim']


# body_data = optimization_points(body_data)
# # 获取非零体素的索引
# indices = np.where(body_data > 0)
# points = np.array([[x * pixdim[1], y * pixdim[2], z * pixdim[3]] for x, y, z in zip(indices[0], indices[1], indices[2])])

# # 转换为 float32 类型
# points = points.astype(np.float32)

# # 创建点云对象
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(points)

# # 使用 alpha 形状算法创建三角网格
# alpha_shape_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, 100)
# alpha_shape_mesh.compute_vertex_normals()

# # 可视化三角网格
# o3d.visualization.draw_geometries([alpha_shape_mesh], window_name='Alpha Shape Mesh', width=800, height=600)