import os

# 指定文件夹路径
folder_path = r'D:\data\CTA\train\A0'

# 获取文件夹中所有文件
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 开始重命名
for index, filename in enumerate(files, start=1):
    # 获取文件扩展名
    name = filename.rsplit('.')[0]
    name = name.rsplit('_')[1]
    file_extension = filename.rsplit('.')[1:]
    # 生成新的文件名
    new_name = f'A0_{name}{'.'}{file_extension[0]}{'.'}{file_extension[1]}'
    # 拼接完整路径
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_name)
    # 重命名文件
    os.rename(old_file, new_file)
    # print(f'Renamed "{old_file}" to "{new_file}"')