"""
测试计算CT上光学小球和FTK500的转换矩阵公式代码

"""
# import numpy as np

# # 假设P_A和P_B是两个坐标系下的点的坐标数组
# P_A = np.array([[-0.028762,-0.013863,1.082586], [-0.104543,-0.027207,1.099426], [-0.052961,-0.043491,1.132218], [0.010597,-0.052946,1.150138]])
# P_B = np.array([[0.09838125, 0.276759375, 0.11120000166], [0.163525112, 0.276120536, 0.09880000147], [0.225013393, 0.275854353, 0.0540000008],[0.1649625, 0.274771875, 0.11520000172]])

# # 计算坐标系A的单位向量
# O_A = P_A.mean(axis=0)
# a_x = (P_A[1] - O_A) / np.linalg.norm(P_A[1] - O_A)
# a_y = (P_A[2] - O_A) / np.linalg.norm(P_A[2] - O_A)
# a_z = np.cross(a_x, a_y)

# # 单位化a_z
# a_z = a_z / np.linalg.norm(a_z)

# # 构造旋转矩阵R
# R = np.column_stack((a_x, a_y, a_z))

# # 计算平移向量T
# T = P_B.mean(axis=0) - np.dot(R, O_A)

# # 构造转换矩阵
# transformation_matrix = np.eye(4)
# transformation_matrix[:3, :3] = R
# transformation_matrix[:3, 3] = T

# # 使用转换矩阵转换点
# points_A = np.array([[x, y, z]])  # 要转换的点在坐标系A中的坐标
# points_B = np.dot(transformation_matrix, np.hstack((points_A, np.ones((1, 1)))))[:3]

# print("转换后的点坐标:", points_B)




import numpy as np

def umeyama(src, dst):
    """
    计算从src点集到dst点集的最优变换矩阵（包括旋转和平移）。
    src和dst是形状为(n_points, 2)或(n_points, 3)的NumPy数组。
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    
    # 中心化点
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    
    # 计算协方差矩阵
    H = np.dot(src_demean.T, dst_demean) / src_demean.shape[0]
    
    # 计算奇异值分解
    U, S, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = np.dot(Vt.T, U.T)
    
    # 计算平移向量
    t = dst_mean.T - np.dot(R, src_mean.T)
    
    # 确保旋转矩阵是正确的
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
        t = dst_mean.T - np.dot(R, src_mean.T)
    
    return R, t

# 假设points_src和points_dst是两个坐标系下的对应点坐标
# points_dst = np.array([[-0.028762,-0.013863,1.082586], [-0.104543,-0.027207,1.099426], [-0.052961,-0.043491,1.132218], [0.010597,-0.052946,1.150138]])
points_dst = np.array([  [0.010597,-0.052946,1.150138],[-0.028762,-0.013863,1.082586],[-0.104543,-0.027207,1.099426],[-0.052961,-0.043491,1.132218]])
points_src = np.array([[0.09838125, 0.276759375, 0.11120000166],[0.1649625, 0.274771875, 0.11520000172], [0.163525112, 0.276120536, 0.09880000147], [0.225013393, 0.275854353, 0.0540000008]])

# points_dst = np.array([[-0.028762,-0.013863,1.082586]])
# points_src = np.array([[0.09838125, 0.276759375, 0.11120000166]])

# 计算变换矩阵
R, t = umeyama(points_src, points_dst)

# 构造最终的变换矩阵
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = R
transformation_matrix[:3, 3] = t

print("旋转矩阵 R:\n", R)
print("平移向量 t:\n", t)
print("变换矩阵:\n", transformation_matrix)


# point_to_transform = np.array([0.09838125, 0.276759375, 0.11120000166, 1]).reshape(4, 1)
point_to_transform = np.array([0.1649625, 0.274771875, 0.11520000172, 1]).reshape(4, 1)


# 应用变换矩阵
transformed_point = np.dot(transformation_matrix, point_to_transform)

# 打印变换后的点
print("变换后的点坐标:", transformed_point[:3].reshape(3))