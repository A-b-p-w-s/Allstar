"""
将同济联影截图数据中各个器官分类成各个文件夹的代码
用于将联影截图数据分类成各个器官的文件夹

"""
import os
import shutil

def divide_png_files(folder_path):
    # 获取文件夹中所有png文件
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    total_png_count = len(png_files)
    
    # 计算每份文件夹中应包含的png文件数量
    files_per_folder = total_png_count // 6
    
    # 创建6个文件夹来保存分割后的png文件
    for i in range(6):
        folder_name = f'part_{i+1}'
        folder_path_new = os.path.join(folder_path, folder_name)
        os.makedirs(folder_path_new, exist_ok=True)
        
        # 将相应数量的png文件移动到新文件夹
        for j in range(files_per_folder):
            file_index = i * files_per_folder + j
            if file_index < total_png_count:
                src_file_path = os.path.join(folder_path, png_files[file_index])
                dst_file_path = os.path.join(folder_path_new, png_files[file_index])
                shutil.move(src_file_path, dst_file_path)
    
    # 处理剩余的png文件（如果总数不能被6整除）
    remaining_files = total_png_count % 6
    for i in range(remaining_files):
        src_file_path = os.path.join(folder_path, png_files[-(i+1)])
        dst_file_path = os.path.join(folder_path, f'part_{i+1}', png_files[-(i+1)])
        shutil.move(src_file_path, dst_file_path)

# 使用示例
folder_path = r'D:\TJ_DATA_JT\DICOMBC_PA0'  # 替换为你的文件夹路径
divide_png_files(folder_path)