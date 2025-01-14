"""
测试计算凸包凹包的代码
"""
# import matplotlib.pyplot as plt
# from scipy.spatial import ConvexHull
# import numpy as np
# import random

# # 生成一个不规则的圆形
# def generate_irregular_circle(center, radius, num_points=100, noise_level=0.1):
#     # 生成圆形的点
#     angles = np.linspace(0, 2 * np.pi, num_points)
#     points = [
#         (center[0] + radius * np.cos(angle) + random.uniform(-noise_level, noise_level),
#          center[1] + radius * np.sin(angle) + random.uniform(-noise_level, noise_level))
#         for angle in angles
#     ]
#     return np.array(points)

# # 生成一个不规则多边形
# def generate_irregular_polygon():
#     # 随机生成多边形的顶点
#     vertices = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(6)]
#     return np.array(vertices)

# # 主函数
# def main():
#     center = (5, 5)
#     radius = 5
#     noise_level = 0.2

#     # irregular_circle_points = generate_irregular_circle(center, radius, noise_level=noise_level)
#     irregular_circle_points= generate_irregular_polygon()
#     hull = ConvexHull(irregular_circle_points)

#     # 输出凸包的顶点索引
#     print("Convex Hull vertices indices:", hull.vertices)
    
#     # 输出凸包的详细坐标
#     print("Convex Hull vertices coordinates:")
#     for i in hull.vertices:
#         print(irregular_circle_points[i])

#     # 绘制图形
#     fig, ax = plt.subplots()
#     ax.plot(irregular_circle_points[:, 0], irregular_circle_points[:, 1], 'o', label='Points')
#     for simplex in hull.simplices:
#         plt.plot(irregular_circle_points[simplex, 0], irregular_circle_points[simplex, 1], 'k-')
#     plt.fill(irregular_circle_points[simplex, 0], irregular_circle_points[simplex, 1], 'k', alpha=0.3)
#     plt.legend()
#     plt.xlim(-1, 11)
#     plt.ylim(-1, 11)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.show()

# if __name__ == "__main__":
#     main()
    
    


import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import random

# 生成一个不规则的圆形
def generate_irregular_circle(center, radius, num_points=100, noise_level=0.1):
    # 生成圆形的点
    angles = np.linspace(0, 2 * np.pi, num_points)
    points = [
        (center[0] + radius * np.cos(angle) + random.uniform(-noise_level, noise_level),
         center[1] + radius * np.sin(angle) + random.uniform(-noise_level, noise_level))
        for angle in angles
    ]
    return np.array(points)

# 生成一个不规则多边形
def generate_irregular_polygon():
    # 随机生成多边形的顶点
    vertices = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(6)]
    return np.array(vertices)

# 计算凹包
def concave_hull(points, ratio=0.3):
    # 计算凸包
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # 使用Delaunay三角剖分
    tri = Delaunay(points)
    
    # 计算每个点到凸包边界的最短距离
    from scipy import spatial
    tree = spatial.cKDTree(points)
    distances = tree.query(hull_points, k=2)[0][:, 1]
    
    # 选择距离凸包边界一定比例内的点作为凹包的点
    concave_points = points[np.where(distances < distances.max() * ratio)]
    
    # 使用Delaunay三角剖分的结果来确定凹包的边界点
    boundary_points = []
    for simplex in tri.simplices:
        for i in range(3):
            pt = points[simplex[i]]
            dist = np.min(np.linalg.norm(points[simplex] - pt, axis=1))
            if dist < distances.max() * ratio:
                boundary_points.append(pt)
    
    # 去重并按顺序排列
    boundary_points = np.array(list(set(map(tuple, boundary_points))))
    boundary_points = boundary_points[np.argsort(np.arctan2(boundary_points[:, 1] - boundary_points[0, 1], boundary_points[:, 0] - boundary_points[0, 0]))]
    
    return boundary_points

# 主函数
def main():
    center = (5, 5)
    radius = 5
    noise_level = 0.2

    irregular_circle_points = generate_irregular_polygon()
    concave_points = concave_hull(irregular_circle_points)

    # 绘制图形
    fig, ax = plt.subplots()
    ax.plot(irregular_circle_points[:, 0], irregular_circle_points[:, 1], 'o', label='Points')
    ax.plot(concave_points[:, 0], concave_points[:, 1], 'x-', label='Concave Hull Points')
    plt.legend()
    plt.xlim(-1, 11)
    plt.ylim(-1, 11)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
    main()