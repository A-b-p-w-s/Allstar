# import numpy as np
# import nibabel as nib

# def generate_spheres(num_spheres, radius, bounds):
#     """生成随机障碍球的位置"""
#     spheres = np.random.uniform(low=bounds[0], high=bounds[1], size=(num_spheres, 3))
#     return spheres

# def check_ray_intersection(ray_origin, ray_direction, spheres, radius):
#     """检查射线与障碍球的交点数量"""
#     # 计算从射线原点到球心的向量
#     oc = spheres - ray_origin
#     # 计算射线方向和球心向量的点积
#     a = np.dot(ray_direction, ray_direction)
#     b = 2 * np.dot(oc, ray_direction)
#     # 计算每个球体的 oc 向量的点积
#     c = np.sum(oc**2, axis=1) - radius**2
#     # 计算判别式
#     discriminant = b**2 - 4*a*c
    
#     # 找到所有可能的交点（判别式非负）
#     mask = discriminant >= 0
    
#     if not np.any(mask):
#         return 0  # 没有交点
    
#     # 计算交点到射线原点的距离
#     sqrt_discriminant = np.sqrt(discriminant[mask])
#     t = (-b[mask] - sqrt_discriminant) / (2*a)
#     t2 = (-b[mask] + sqrt_discriminant) / (2*a)
    
#     # 找到有效的交点（t >= 0）
#     valid_t = (t >= 0) | (t2 >= 0)
#     intersections = np.sum(valid_t)
    
#     return intersections

# # 设置参数
# num_spheres = 10000000
# radius = 1
# bounds = [-100, 100]  # 障碍球生成的边界

# # 生成障碍球
# # spheres = generate_spheres(num_spheres, radius, bounds)

# # 读取.nii.gz文件
# nii_img = nib.load(r'C:\Users\allstar\Desktop\body_model\bone\bone.nii')  # 替换为你的文件路径
# # C:\Users\allstar\Desktop\body_model\bone\bone.nii
# # C:\Users\allstar\Desktop\body_model\new_tumour\tumour.nii.gz
# nii_data = nii_img.get_fdata()

# # 获取所有像素值为1的点的索引
# indices = np.where(nii_data == 1)

# # 创建一个空的三维数组来存储这些点
# points = np.zeros((len(indices[0]), 3), dtype=np.int16)

# # 将索引转换为三维坐标
# for i, (idx1, idx2, idx3) in enumerate(zip(indices[0], indices[1], indices[2])):
#     points[i] = [idx1, idx2, idx3]

# spheres = points

# # 设置点源和几条射线
# ray_origin = np.array([271 ,186 ,153])
# # ray_directions = [
# #     np.array([1, 0, 0]),
# #     np.array([0, 1, 0]),
# #     np.array([0, 0, 1]),
# #     np.array([1, 1, 1]),
# #     # 可以添加更多射线方向
# # ]

# ray_directions=[[265.6         ,69.6         ,66.        ],
#              [158.4         ,66.4         ,71.        ],
#              [362.28571429  ,67.85714286  ,86.5       ],
#              [263.28571429  ,67.42857143 ,142.5       ]]
# # ray_directions = points

# # 计算每条射线碰到障碍球的个数
# for i, direction in enumerate(ray_directions):
#     direction = direction / np.linalg.norm(direction)  # 单位化方向向量
#     count = check_ray_intersection(ray_origin, direction, spheres, radius)
#     print(f"Ray {i+1} intersections: {count}")



# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def generate_spheres(num_spheres, radius):
#     """生成随机障碍球的中心点"""
#     centers = np.random.rand(num_spheres, 3) * 10  # 假设在10x10x10的立方体内生成
#     return centers

# def generate_rays(source, num_rays, direction_range=(0, 2*np.pi)):
#     """生成从点源发出的射线"""
#     directions = np.random.uniform(direction_range[0], direction_range[1], num_rays)
#     rays = []
#     for direction in directions:
#         ray_direction = np.array([np.cos(direction), np.sin(direction), 0])
#         rays.append(np.concatenate([source, ray_direction]))
#     return np.array(rays)

# def check_intersection(ray, sphere, radius):
#     """检查射线与球是否相交"""
#     source = ray[:3]
#     direction = ray[3:]
#     center = sphere
#     A = np.dot(direction, direction)
#     B = 2 * np.dot(direction, center - source)
#     C = np.dot(center - source, center - source) - radius**2

#     discriminant = B**2 - 4*A*C
#     if discriminant < 0:
#         return 0  # 无交点
#     elif discriminant == 0:
#         return 1  # 相切
#     else:
#         return 2  # 相交两次

# def main():
#     num_spheres = 100
#     num_rays = 10
#     radius = 1

#     centers = generate_spheres(num_spheres, radius)
#     rays = generate_rays(source=np.array([271 ,186 ,153]), num_rays=num_rays)

#     results = []
#     for ray in rays:
#         count = 0
#         for center in centers:
#             count += check_intersection(ray, center, radius)
#         results.append(count)

#     # 可视化
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for center in centers:
#         u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#         x = center[0] + radius*np.cos(u)*np.sin(v)
#         y = center[1] + radius*np.sin(u)*np.sin(v)
#         z = center[2] + radius*np.cos(v)
#         ax.plot_wireframe(x, y, z, color='b', alpha=0.1)

#     ax.scatter([0] * num_rays, [0] * num_rays, [0] * num_rays, color='r', label='Source')
#     for ray in rays:
#         start_point = ray[:3]
#         end_point = start_point + 10 * ray[3:]  # 射线长度为10倍方向向量
#         ax.quiver(*start_point, *ray[3:], length=10, normalize=False)
#     plt.legend()
#     plt.show()

#     print("Ray intersections:", results)

# if __name__ == "__main__":
#     main()
    




# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import nibabel as nib

# # 参数设置
# num_obstacles = 100  # 障碍球数量
# radius = 1  # 障碍球半径
# source = np.array([271 ,186 ,153])  # 点源位置
# targets = np.array([
#             [265.6         ,69.6         ,66.        ],
#              [158.4         ,66.4         ,71.        ],
#              [362.28571429  ,67.85714286  ,86.5       ],
#              [263.28571429  ,67.42857143 ,142.5       ]
# ])

# # 读取.nii.gz文件
# nii_img = nib.load(r'C:\Users\allstar\Desktop\body_model\bone\bone.nii')  # 替换为你的文件路径
# nii_data = nii_img.get_fdata()

# # 获取所有像素值为1的点的索引
# indices = np.where(nii_data == 1)

# # 使用 NumPy 的向量化操作来创建障碍点的数组
# points = np.vstack((indices,)).T

# obstacles = np.array(points)  # 直接使用 targets 作为障碍点

# # 计算射线与障碍球的交点
# def ray_intersects(obstacle, source, direction):
#     # 计算射线与球的交点
#     oc = obstacle - source
#     a = np.dot(direction, direction)
#     b = 2 * np.dot(direction, oc)
#     c = np.dot(oc, oc) - radius**2
#     discriminant = b**2 - 4*a*c
#     if discriminant < 0:
#         return 0  # 无交点
#     else:
#         t1 = (-b - np.sqrt(discriminant)) / (2*a)
#         t2 = (-b + np.sqrt(discriminant)) / (2*a)
#         return int(t1 > 0 or t2 > 0)

# # 计算每条射线碰到的障碍球数量
# intersection_counts = []
# for target in targets:
#     direction = (source - target) / np.linalg.norm(source - target)  # 单位方向向量
#     count = 0
#     for obstacle in obstacles:
#         if ray_intersects(obstacle, source, direction):
#             count += 1
#     intersection_counts.append(count)
    
# # 输出每条射线碰到的障碍球数量
# print(intersection_counts)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# # 可视化
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制障碍球
# for obstacle in obstacles:
#     u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#     x = obstacle[0] + radius * np.cos(u) * np.sin(v)
#     y = obstacle[1] + radius * np.sin(u) * np.sin(v)
#     z = obstacle[2] + radius * np.cos(v)
#     ax.plot_surface(x, y, z, color='b', alpha=0.3)

# # 绘制射线
# for i, target in enumerate(targets):
#     ax.plot([source[0], target[0]], [source[1], target[1]], [source[2], target[2]], 'ro-', label=f'Ray to Target {i+1}')

# # 显示
# ax.scatter(*source, color='g', label='Source')
# ax.scatter(*targets.T, color='m', label='Targets')
# ax.legend()
# plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
from tqdm import tqdm

# 参数设置
radius = 1  # 障碍球半径
source = np.array([271 ,186 ,153])  # 点源位置
targets = np.array([
            [265.6         ,69.6         ,66.        ],
            [158.4         ,66.4         ,71.        ],
            [362.28571429  ,67.85714286  ,86.5       ],
            [263.28571429  ,67.42857143 ,142.5       ]
])


# 找到x、y、z轴上的最小值和最大值
min_x, min_y, min_z = targets.min(axis=0)
max_x, max_y, max_z = targets.max(axis=0)
x_values = np.arange(min_x, max_x, 10)
y_values = np.arange(min_y, max_y, 10)
z_values = np.arange(min_z, max_z, 10)

# 使用meshgrid创建三维网格
xx, yy, zz = np.meshgrid(x_values, y_values, z_values, indexing='ij')

# 将网格展平为一维数组
points_in_box = np.array(np.meshgrid(x_values, y_values, z_values, indexing='ij')).T.reshape(-1, 3)

targets = points_in_box

marker_img=nib.load(r"C:\Users\allstar\Desktop\body_model\Marker\processed_image_abs.nii.gz")
nii_data3 = marker_img.get_fdata()

# 获取所有像素值为1的点的索引
indices3 = np.where(nii_data3 > 0)

# 使用 NumPy 的向量化操作来创建障碍点的数组
points3 = np.vstack((indices3,)).T

boundary_img=nib.load(r"C:\Users\allstar\Desktop\body_model\boundary_new\boundary.nii.gz")
nii_data2 = boundary_img.get_fdata()

# 获取所有像素值为1的点的索引
indices2 = np.where(nii_data2 > 0)

# 使用 NumPy 的向量化操作来创建障碍点的数组
points2 = np.vstack((indices2,)).T

# 读取.nii.gz文件
nii_img = nib.load(r'C:\Users\allstar\Desktop\body_model\bone\bone.nii')  # 替换为你的文件路径
nii_data = nii_img.get_fdata()

# 获取所有像素值为1的点的索引
indices = np.where(nii_data > 0)

# 使用 NumPy 的向量化操作来创建障碍点的数组
points = np.vstack((indices,)).T

obstacles = np.concatenate((points, points2,points3), axis=0)

pixdim = nii_img.header['pixdim']

print(len(obstacles))


# obstacles = np.array(points)  # 直接使用 targets 作为障碍点

# targets = points[0:1000]

# 计算射线与障碍球的交点
def ray_intersects(obstacles, source, directions):
    # 计算射线与球的交点
    oc = obstacles - source
    a = np.sum(directions**2, axis=1)
    b = 2 * np.sum(directions * oc, axis=1)
    c = np.sum(oc**2, axis=1) - radius**2
    discriminant = b**2 - 4*a*c
    
    #----------------------------------------------------------------------------------------------------
    # non_negative_discriminant = discriminant >= 0
    
    # # 计算t1和t2，只考虑非负的解
    # t1 = (-b - np.sqrt(discriminant)) / (2*a)
    # t2 = (-b + np.sqrt(discriminant)) / (2*a)
    
    # # 选择有效的t值
    # t = np.maximum(t1, t2)
    # t = t[non_negative_discriminant]  # 只考虑非负的discriminant
    #-----------------------------------------------------------------------------------------------------
    
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
zero_indices  = np.where( np.array(intersection_counts) == 0)
distances = np.linalg.norm(targets - source, axis=1)

distances_zero = distances[zero_indices]
print(len(distances_zero[distances_zero<150]))
print(len(distances[zero_indices]))
print(np.argsort(distances[zero_indices])[0])
print(targets[np.argsort(distances[zero_indices])[0]][0],targets[np.argsort(distances[zero_indices])[0]][1],targets[np.argsort(distances[zero_indices])[0]][2])
print((targets[np.argsort(distances[zero_indices])[0]][0])*pixdim[1],(nii_data.shape[0]-targets[np.argsort(distances[zero_indices])[0]][1])*pixdim[2],(nii_data.shape[2]-targets[np.argsort(distances[zero_indices])[0]][2])*pixdim[3])
