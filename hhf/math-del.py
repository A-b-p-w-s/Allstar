"""
测试计算旋转角度的代码

"""

import math


# 定义两个点的三维坐标
point_a = (98.38125 ,  276.759375 ,  111.20000166)  # 原点
point_b = (225.01339286 , 275.85435268 , 98.80000147)  # 另一个点的坐标，需要替换x, y, z为实际值

# 计算向量AB
vector_ab = (point_b[0] - point_a[0], point_b[1] - point_a[1], point_b[2] - point_a[2])

# 规范化向量
length = math.sqrt(vector_ab[0]**2 + vector_ab[1]**2 + vector_ab[2]**2)
if length == 0:
    raise ValueError("The vector is zero, cannot calculate angles.")
vector_ab = (vector_ab[0] / length, vector_ab[1] / length, vector_ab[2] / length)

# 计算欧拉角
pitch = math.asin(-vector_ab[2])  # 俯仰角，绕x轴
yaw = math.atan2(vector_ab[1], vector_ab[0])  # 偏航角，绕y轴
roll = math.atan2(vector_ab[2] * math.cos(pitch), vector_ab[0] * math.cos(pitch))  # 翻滚角，绕z轴

# 将弧度转换为度
pitch_degrees = math.degrees(pitch)
yaw_degrees = math.degrees(yaw)
roll_degrees = math.degrees(roll)

# 打印结果
print(f"Pitch (rotation around x-axis): {pitch_degrees} degrees")
print(f"Yaw (rotation around y-axis): {yaw_degrees} degrees")
print(f"Roll (rotation around z-axis): {roll_degrees} degrees")