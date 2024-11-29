import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个新的图和一个子图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# 定义一些点的坐标
points = [
    (-0.028762,-0.013863,1.082586),  # 点1
    ( -0.104543,-0.027207,1.099426), # 点2
    (-0.052961,-0.043491,1.132218), # 点3
    (0.010597,-0.052946,1.150138), # 点4
    # 可以继续添加更多的点
]

# 定义一些点的坐标
points2 = [
    (-0.01815345537771096, -0.002718010492624379, 1.063409270712447),  # 点1
    ( -0.04006736378225534, -0.03540868647951295, 1.116785212841954), # 点2
    (-0.09343457624915187, -0.0621477851004657, 1.163953855521265), # 点3
    (-0.02401360459088187, -0.03723251792739707, 1.120219660924334), # 点4
    # 可以继续添加更多的点
]
# 绘制这些点
for point in points:
    ax.scatter(*point, color='red')  # 使用解包操作符*来传递点的坐标
    
    # 绘制这些点
for point in points2:
    print(*point)
    ax.scatter(*point, color='blue')  # 使用解包操作符*来传递点的坐标

# 设置坐标轴的标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.show()