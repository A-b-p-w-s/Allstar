"""
测试计算最近路劲点的坐标的代码
"""
# import matplotlib.pyplot as plt
# from shapely.geometry import Polygon, Point, LineString
# import random

# # 生成一个不规则多边形
# def generate_irregular_polygon():
#     # 随机生成多边形的顶点
#     vertices = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(6)]  # 增加顶点数以确保多边形更不规则
#     return Polygon(vertices)

# # 在多边形内部随机选择一个点
# def random_point_inside_polygon(polygon):
#     while True:
#         x = random.uniform(0, 10)
#         y = random.uniform(0, 10)
#         point = Point(x, y)
#         if polygon.contains(point):
#             return point

# # 计算点到多边形边的垂直点
# def nearest_perpendicular_point(point, polygon):
#     nearest_point = None
#     min_distance = float('inf')
#     for i in range(len(polygon.exterior.coords)):
#         start_point = polygon.exterior.coords[i]
#         end_point = polygon.exterior.coords[(i + 1) % len(polygon.exterior.coords)]
#         edge = LineString([start_point, end_point])
#         perpendicular_point = edge.project(point)
#         if perpendicular_point < min_distance:
#             min_distance = perpendicular_point
#             nearest_point = edge.interpolate(perpendicular_point)
#     return nearest_point

# # 主函数
# def main():
#     polygon = generate_irregular_polygon()
#     point_inside = random_point_inside_polygon(polygon)
#     nearest_pt = nearest_perpendicular_point(point_inside, polygon)

#     # 绘制图形
#     fig, ax = plt.subplots()
#     polygon_coords = list(polygon.exterior.coords)  # 获取多边形的坐标并转换为列表
#     polygon_patch = plt.Polygon(polygon_coords, closed=True, edgecolor='blue', facecolor='none', alpha=0.5, lw=2)
#     ax.add_patch(polygon_patch)
#     ax.plot(*point_inside.xy, 'ro')  # 绘制内部点
#     ax.plot(*nearest_pt.xy, 'go')  # 绘制最近垂直点
#     plt.xlim(0, 10)
#     plt.ylim(0, 10)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.show()

#     print(f"内部点坐标: {point_inside}")
#     print(f"最近垂直点坐标: {nearest_pt}")

# if __name__ == "__main__":
#     main()

# =====================================================================================================================================================================================

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
import random
from scipy.spatial import ConvexHull



# 生成一个不规则多边形
def generate_irregular_polygon():
    # 随机生成多边形的顶点
    vertices = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(6)]
    return vertices

# 在多边形内部随机选择一个点
def random_point_inside_polygon(polygon):
    while True:
        point = Point(random.uniform(0, 10), random.uniform(0, 10))
        if Polygon(polygon).contains(point):
            return point

# 计算点到多边形边的最近垂直点
def nearest_perpendicular_point(point, polygon):
    nearest_distance = float('inf')
    nearest_point = None
    for i in range(len(polygon.exterior.coords)):
        start_point = polygon.exterior.coords[i]
        end_point = polygon.exterior.coords[(i + 1) % len(polygon.exterior.coords)]
        edge = LineString([start_point, end_point])
        start, end = nearest_points(point, edge)
        if point.distance(end) < nearest_distance:
            nearest_distance = point.distance(end)
            nearest_point = end
    return nearest_point

# 主函数
def main():
    polygon = generate_irregular_polygon()
    hull = ConvexHull(polygon)
    point_inside = random_point_inside_polygon(polygon)
    
    hull_a=Polygon([polygon[i] for i in hull.vertices])
    
    nearest_pt = nearest_perpendicular_point(point_inside, hull_a)

    # 绘制图形
    fig, ax = plt.subplots()
    polygon_patch = plt.Polygon(list(Polygon(polygon).exterior.coords), closed=True, edgecolor='blue', facecolor='none', alpha=0.5, lw=2)
    polygon_patch2 = plt.Polygon(list(hull_a.exterior.coords), closed=True, edgecolor='yellow', facecolor='none', alpha=0.8, lw=2)
    ax.add_patch(polygon_patch)
    ax.add_patch(polygon_patch2)
    ax.plot(*point_inside.xy, 'ro')  # 绘制内部点
    ax.plot(*nearest_pt.xy, 'go')  # 绘制最近垂直点
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    print(f"内部点坐标: {point_inside}")
    print(f"最近垂直点坐标: {nearest_pt}")

if __name__ == "__main__":
    main()