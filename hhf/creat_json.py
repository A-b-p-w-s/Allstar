"""
创建nnunet所需要的json文件代码
用于生成nnunet，dataset中所需要的json文件
"""
import json

# 创建一个字典，这将是我们的JSON数据
data = {
    "name": "John Doe",
    "age": 30,
    "city": "New York",
    "hasChildren": False,
    "titles": ["engineer", "programmer"]
}

# 指定JSON文件的名称
filename = r'C:\Users\allstar\Desktop\marker\data.json'

# 将字典转换为JSON格式并写入文件
with open(filename, 'w') as f:
    json.dump(data, f, indent=4)  # indent参数用于美化输出

print(f"JSON data has been written to {filename}")