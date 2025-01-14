"""
测试路劲规划中最近路劲点的坐标的代码
"""
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
import random
from scipy.spatial import ConvexHull
import nibabel as nib
import numpy as np
import cv2


# nii_image3 = nib.load(r'C:\Users\allstar\Desktop\test\p\0.nii.gz')  # 替换为你的文件路径
nii_image3 = nib.load(r'C:\Users\allstar\Desktop\body_model\body_model\Segmentation.nii')  # 替换为你的文件路径
body_data = nii_image3.get_fdata()

# aaa=body_data[:,:,60].astype(np.uint8)*255
aaa=(body_data[:,:,60]*255).astype(np.uint8)

# 使用Canny算法进行边缘检测
edges = cv2.Canny(aaa, threshold1=50, threshold2=150)

indices = np.where(edges >0)
coordinates = [(x, y) for x, y in zip(indices[0], indices[1])]


aaa_hull_X=[]
aaa_hull_Y=[]

# 计算点到多边形边的最近垂直点
def nearest_perpendicular_point(point, polygon,abc):
    nearest_distance = float('inf')
    nearest_point = None
    for i in range(len(polygon.exterior.coords)):
        start_point = polygon.exterior.coords[i]
        end_point = polygon.exterior.coords[(i + 1) % len(polygon.exterior.coords)]
        edge = LineString([start_point, end_point])
        start, end = nearest_points(point, edge)
        aaa_hull_X.append(end.coords[0][0])
        aaa_hull_Y.append(end.coords[0][1])
        abc = end
        if point.distance(end) < nearest_distance:
            nearest_distance = point.distance(end)
            nearest_point = end
    return nearest_point,abc

# 主函数
def main():
    polygon = coordinates
    hull = ConvexHull(polygon)
    # point_inside = Point(271 ,186)
    point_inside = Point(168 ,229)
    
    hull_a=Polygon([polygon[i] for i in hull.vertices])
    
    
    abc = Point(0,0)
    nearest_pt,abc = nearest_perpendicular_point(point_inside, hull_a,abc)
    
    origin = point_inside.coords[0]
    target = nearest_pt.coords[0]
    
    # 直线的斜率
    slope = (target[1] - origin[1]) / (target[0] - origin[0]) if target[0] != origin[0] else float('inf')
    # 直线的截距
    # intercept = origin[1] - slope * origin[0] if slope != float('inf') else 0

    # 3. 计算垂直于该直线的垂线的方程
    # 垂直线的斜率是原直线斜率的负倒数
    perpendicular_slope = -1 / slope if slope != 0 else 0
    # 通过目标点的垂直线的截距
    perpendicular_intercept = target[1] - perpendicular_slope * target[0]
    
    x_perp_vals = np.linspace(target[0] - 50, target[0] + 50, 2)
    y_perp_vals = perpendicular_slope * x_perp_vals + perpendicular_intercept
    
    
    # 绘制图形
    fig, ax = plt.subplots()
    polygon_patch = plt.Polygon(list(Polygon(polygon).exterior.coords), closed=True, edgecolor='blue', facecolor='none', alpha=0.5, lw=2)
    polygon_patch2 = plt.Polygon(list(hull_a.exterior.coords), closed=True, edgecolor='yellow', facecolor='none', alpha=0.8, lw=2)
    ax.add_patch(polygon_patch)
    ax.add_patch(polygon_patch2)
    
    plt.plot(x_perp_vals, y_perp_vals, 'b', label='Perpendicular Line', lw=2) #绘制肝膜水平线
    ax.plot(*point_inside.xy, 'ro')  # 绘制内部点
    ax.plot(*nearest_pt.xy, 'go')  # 绘制最近垂直点
    
    # ax.plot(*abc.xy, 'bo')
    
    # ax.plot(aaa_hull_X[0],aaa_hull_Y[0], 'ro')
    # ax.plot(aaa_hull_X[1],aaa_hull_Y[1], 'go') 
    # ax.plot(aaa_hull_X[2],aaa_hull_Y[2], 'bo')
    

    
    plt.xlim(0, 512 )
    plt.ylim(0, 512 )
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    
    print(f"内部点坐标: {point_inside}")
    print(f"最近垂直点坐标: {nearest_pt}")


if __name__ == "__main__":
    main()