"""
测试绘制三维空间中线和面的代码
"""
# import numpy as np
# import matplotlib.pyplot as plt

# # 设置随机种子
# np.random.seed(0)

# # 1. 随机生成两个点
# origin = np.random.rand(2) * 10
# target = np.random.rand(2) * 10

# # 2. 计算两点之间的直线方程
# # 直线的斜率
# slope = (target[1] - origin[1]) / (target[0] - origin[0]) if target[0] != origin[0] else float('inf')
# # 直线的截距
# intercept = origin[1] - slope * origin[0] if slope != float('inf') else 0

# # 3. 计算垂直于该直线的垂线的方程
# # 垂直线的斜率是原直线斜率的负倒数
# perpendicular_slope = -1 / slope if slope != 0 else 0
# # 通过目标点的垂直线的截距
# perpendicular_intercept = target[1] - perpendicular_slope * target[0]

# # 4. 可视化
# plt.figure(figsize=(8, 8))

# # 绘制原点和目标点
# plt.plot(origin[0], origin[1], 'ro', label='Origin')
# plt.plot(target[0], target[1], 'bo', label='Target')

# # 绘制直线
# x_vals = np.linspace(min(origin[0], target[0]) - 1, max(origin[0], target[0]) + 1, 10)
# y_vals = slope * x_vals + intercept if slope != float('inf') else np.ones(10) * intercept
# plt.plot(x_vals, y_vals, 'r', label='Line')

# # 绘制垂线
# x_perp_vals = np.linspace(min(origin[0], target[0]) - 1, max(origin[0], target[0]) + 1, 10)
# y_perp_vals = perpendicular_slope * x_perp_vals + perpendicular_intercept
# plt.plot(x_perp_vals, y_perp_vals, 'b', label='Perpendicular Line')

# # 设置图例
# plt.legend()

# # 设置坐标轴比例相等
# plt.axis('equal')

# # 设置标题
# plt.title('Line and Perpendicular Line')

# # 显示图表
# plt.grid(True)
# plt.show()



#==========================================================================================================================================================================================================


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 参数设置
# radius = 1  # 球体半径
# num_attempts = 100  # 生成随机点的尝试次数

# # 生成球体内的随机点
# def generate_random_point_inside_sphere(radius):
#     while True:
#         x, y, z = np.random.uniform(-radius, radius, 3)
#         if x**2 + y**2 + z**2 <= radius**2:
#             return x, y, z

# # 生成球体外的随机点
# def generate_random_point_outside_sphere(radius, num_attempts):
#     for _ in range(num_attempts):
#         x, y, z = np.random.uniform(-2*radius, 2*radius, 3)
#         if x**2 + y**2 + z**2 > radius**2:
#             return x, y, z

# # 计算两点之间的连线与球体表面的夹角
# def calculate_angle(p1, p2, center=(0,0,0), radius=1):
#     # 向量
#     v1 = np.array(p1) - center
#     v2 = np.array(p2) - center
#     # 单位向量
#     u1 = v1 / np.linalg.norm(v1)
#     u2 = v2 / np.linalg.norm(v2)
#     # 点积
#     dot_product = np.dot(u1, u2)
#     # 夹角
#     angle = np.arccos(dot_product)
#     return np.degrees(angle)

# # 生成点
# point_inside = generate_random_point_inside_sphere(radius)
# point_outside = generate_random_point_outside_sphere(radius, num_attempts)

# # 计算角度
# angle = calculate_angle(point_inside, point_outside)

# # 可视化
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制球体
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# x = radius * np.cos(u) * np.sin(v)
# y = radius * np.sin(u) * np.sin(v)
# z = radius * np.cos(v)
# ax.plot_wireframe(x, y, z, color='b', alpha=0.1)

# # 绘制点
# ax.scatter(*point_inside, color='r', label='Inside Point')
# ax.scatter(*point_outside, color='g', label='Outside Point')

# # 绘制连线
# ax.plot([point_inside[0], point_outside[0]], [point_inside[1], point_outside[1]], [point_inside[2], point_outside[2]], color='k')

# # 设置图例和标签
# ax.legend()
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # 显示图形
# plt.title(f'Angle between line and sphere surface: {angle:.2f} degrees')
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 参数设置
major_radius = 3  # 环面主半径
minor_radius = 1  # 环面次半径

# 定义旋转环面的参数方程
def torus(u, v):
    u, v = np.radians(u), np.radians(v)
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)
    return x, y, z

# 生成曲面内的随机点
def generate_random_point_inside_torus(major_radius, minor_radius):
    while True:
        theta, phi = np.random.uniform(0, 2 * np.pi, 2)
        r = minor_radius + np.random.uniform(-0.5, 0.5)  # 确保点在曲面内部
        if (r - minor_radius)**2 + major_radius**2 * np.sin(theta)**2 <= major_radius**2:
            return torus(theta, phi)

# 生成曲面外的随机点
def generate_random_point_outside_torus(major_radius, minor_radius, num_attempts):
    for _ in range(num_attempts):
        x, y, z = np.random.uniform(-4*major_radius, 4*major_radius, 3)
        theta = np.arctan2(y, x)
        phi = np.arcsin(z / np.sqrt(x**2 + y**2))
        r = np.sqrt((x - major_radius * np.cos(theta))**2 + (y - major_radius * np.sin(theta))**2) + minor_radius
        if r > minor_radius:
            return x, y, z

# 计算两点之间的连线与曲面的夹角
def calculate_angle(p1, p2):
    v = np.array(p2) - np.array(p1)
    v_norm = v / np.linalg.norm(v)
    # 计算法向量，这里简化为使用点到原点的向量
    normal = np.array([2*p1[0], 2*p1[1], 2*p1[2]])
    u = normal / np.linalg.norm(normal)
    dot_product = np.dot(u, v_norm)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

# 生成点
point_inside = generate_random_point_inside_torus(major_radius, minor_radius)
point_outside = generate_random_point_outside_torus(major_radius, minor_radius, 100)

# 计算角度
angle = calculate_angle(point_inside, point_outside)

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制旋转环面
u, v = np.mgrid[0:2*np.pi:50j, 0:2*np.pi:50j]
x, y, z = torus(u, v)
ax.plot_surface(x, y, z, color='b', alpha=0.3)

# 绘制点
ax.scatter(*point_inside, color='r', label='Inside Point')
ax.scatter(*point_outside, color='g', label='Outside Point')

# 绘制连线
ax.plot([point_inside[0], point_outside[0]], [point_inside[1], point_outside[1]], [point_inside[2], point_outside[2]], color='k')

# 设置图例和标签
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.title(f'Angle between line and torus surface: {angle:.2f} degrees')
plt.show()