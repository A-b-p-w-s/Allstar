"""
测试将stl文件转换为三维数据的代码
"""
r'C:\Users\allstar\Desktop\dataset\龙潇STL\龙潇STL\a\动脉.stl'  # 替换为您的STL文件路径
import numpy as np
from stl import mesh

# 读取STL文件并获取其三维数据
def load_stl_data(file_path):
    # 加载STL文件
    stl_mesh = mesh.Mesh.from_file(file_path)
    return stl_mesh.vectors

# 显示STL文件的三维数组数据
def display_stl_data(file_path):
    # 读取STL数据
    stl_data = load_stl_data(file_path)
    stl_data = stl_data.reshape(-1, 3) 
    print(stl_data)
        
# 运行函数，读取指定的STL文件
file_path = r'C:\Users\allstar\Desktop\dataset\龙潇STL\龙潇STL\a\动脉.stl'  # 替换为你的STL文件路径
display_stl_data(file_path)