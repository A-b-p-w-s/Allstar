"""
测试计算每个点到线段的距离的代码
"""
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial import cKDTree

# # 设置随机种子以获得可重复的结果
# np.random.seed(42)

# # 定义源点和目标点
# source = np.array([0, 0, 0])
# b = np.array([10, 10, 0])
# r=1

# # 生成100个障碍点，每个点的半径为2
# # obstacles = np.random.uniform(-10, 10, (100, 3)) + r * np.random.randn(100, 3)

# obstacles = np.array([[10, 10, 0], [10, 12, 0]])

# # 构建障碍点的KD树以加速查询
# tree = cKDTree(obstacles)

# # 定义线段c_link
# c_link = np.vstack((source, b))

# # 检查线段与障碍点之间的交点
# distances, indices = tree.query(c_link, k=2)  # 查询每个点的最近两个障碍点
# intersections = np.any(np.linalg.norm(obstacles[indices] - c_link, axis=1) < (r*2), axis=1)  # 检查距离是否小于障碍点半径的两倍

# # 可视化
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制障碍点
# ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], c='r', label='Obstacles')

# # 绘制线段
# ax.plot(c_link[:, 0], c_link[:, 1], c_link[:, 2], c='b', label='Line Segment')

# # 标记相交的障碍点
# for i in np.where(intersections)[0]:
#     ax.scatter(obstacles[i, 0], obstacles[i, 1], obstacles[i, 2], c='g', s=100, label='Intersecting Obstacle')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# plt.show()



# ===========================================================================================================================================================================================
# ===========================================================================================================================================================================================
# ===========================================================================================================================================================================================


# import numpy as np

# def point_to_line_segment_distance_vectorized(points, A, B):
    # # 计算向量
    # AP = points - A
    # AB = B - A
    
    # # 计算点积
    # dot_product = np.sum(AP * AB, axis=1)
    
    # # 计算AB的模长平方
    # magnitude_squared_AB = np.sum(AB * AB)
    
    # # 计算标量投影
    # scalar_projection = dot_product / magnitude_squared_AB
    
    # # 计算投影点P'
    # P_prime = A + scalar_projection[:, np.newaxis] * AB
    
    # # 计算P'到points的向量
    # PP_prime = points - P_prime
    
    # # 计算P'到points的距离
    # distance_PP_prime = np.linalg.norm(PP_prime, axis=1)
    
    # # 计算P到A和P到B的距离
    # distance_PA = np.linalg.norm(points - A, axis=1)
    # distance_PB = np.linalg.norm(points - B, axis=1)
    
    # # 检查P'是否在线段AB上
    # mask = (scalar_projection >= 0) & (scalar_projection <= 1)
    # distances = np.where(mask, distance_PP_prime, np.minimum(distance_PA, distance_PB))
    
    # return distances

# # 示例点和线段
# A = np.array([0, 0, 0])
# B = np.array([10, 10, 10])

# # 生成100个随机点
# points = np.random.uniform(-10, 10, (100000000, 3))

# # 计算每个点到线段的距离
# distances = point_to_line_segment_distance_vectorized(points, A, B)

# # 打印距离
# print(distances)


# ===========================================================================================================================================================================================
# ===========================================================================================================================================================================================
# ===========================================================================================================================================================================================



import numpy as np

def point_to_line_segment_distance_vectorized(points, A, B):
    # 计算向量
    AB = B - A
    AP = points - A
    BP = points - B
    
    # 计算AB的模长平方
    magnitude_squared_AB = np.sum(AB**2)
    
    # 计算点积
    dot_product_AB_AP = np.sum(AB * AP, axis=1)  # 使用axis=1进行逐行点积
    dot_product_AB_BP = np.sum(AB * BP, axis=1)  # 使用axis=1进行逐行点积
    
    # 计算标量投影
    t = dot_product_AB_AP / magnitude_squared_AB
    u = dot_product_AB_BP / magnitude_squared_AB
    
    # 计算P'到points的距离
    distance_PP_prime = np.sum((AP - t[:, np.newaxis] * AB) ** 2, axis=1)  # 使用axis=1计算平方和
    
    # 计算P到A和P到B的距离
    distance_PA = np.sum(AP**2, axis=1)
    distance_PB = np.sum(BP**2, axis=1)
    
    # 检查P'是否在线段AB上，并计算最终距离
    distances = np.where((t >= 0) & (t <= 1), np.sqrt(distance_PP_prime), np.minimum(np.sqrt(distance_PA), np.sqrt(distance_PB)))
    
    return distances 

# 示例点和线段
A = np.array([0, 0, 0])
B = np.array([10, 10,10])

# 生成100个随机点
# points = np.random.uniform(-10, 10, (10, 3))

points = np.array([[10,10,10],[1,1,1],[10,10,0]])


# 计算每个点到线段的距离
distances = point_to_line_segment_distance_vectorized(points, A, B)

print(distances)



# [[ 38 101 343]
#  [ 38 101 344]
#  [ 38 101 345]
#  ...
#  [172 132 136]
#  [172 132 137]
#  [172 132 138]]