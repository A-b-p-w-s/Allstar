"""
测试计算生成图像近似轮廓的代码
"""
# import cv2
# import numpy as np
# import nibabel as nib

# # 设置随机种子以获得可重复的结果
# np.random.seed(42)

# # 创建一个空白的画布（黑色图像）
# height, width = 512, 512
# img = np.zeros((height, width, 3), dtype=np.uint8)

# # nii_image3 = nib.load(r'C:\Users\allstar\Desktop\test\p\0.nii.gz')  # 替换为你的文件路径
# nii_image3 = nib.load(r'C:\Users\allstar\Desktop\body_model\body_model\Segmentation.nii')  # 替换为你的文件路径
# body_data = nii_image3.get_fdata()

# gray=(body_data[:,:,153]*255).astype(np.uint8)

# source_tumour = np.array([186 ,271])

# targets = np.array([69,265])

# # 二值化
# _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# # 查找轮廓
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 绘制原始轮廓和近似轮廓
# cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# num_p=0
# if contours:
#     for c in contours:
#         # 近似轮廓
#         epsilon = 0.005 * cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, epsilon, True)
#         cv2.polylines(img, [approx], True, (0, 0, 255), 2)
        
#         cv2.circle(img, approx[0][0], 5, (255, 0, 0), -1)   
        
#         num_p+=len(approx)

# print(num_p)

# cv2.circle(img, source_tumour, 5, (255, 0, 0), -1)
# cv2.circle(img, targets, 5, (255, 255, 0), -1)

# # 显示图像
# cv2.imshow('Irregular Shape with Approximated Contour', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import cv2
import numpy as np
import nibabel as nib
import math

def calculate_angle(p1, p2, p3, p4):
    # 计算向量
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p4[0] - p3[0], p4[1] - p3[1])
    
    # 计算点积
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # 计算向量的模
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # 计算夹角的余弦值
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    
    # 计算夹角（以度为单位）
    angle = math.degrees(math.acos(cos_angle))
    
    return angle

def line_segment_intersection(line1, line2):
    # 线段定义为 (x1, y1)-(x2, y2)
    x1, y1 = line1[0], line1[1]
    x2, y2 = line1[2], line1[3]
    x3, y3 = line2[0], line2[1]
    x4, y4 = line2[2], line2[3]
    
    # 计算叉乘
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # 线段平行或重合，无交点

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        # 线段相交，计算交点
        # return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return True
    else:
        return None  # 线段不相交


# 创建一个空白的画布（黑色图像）
height, width = 512, 512
img = np.zeros((height, width, 3), dtype=np.uint8)

# 加载NIfTI图像
# nii_image3 = nib.load(r'C:\Users\allstar\Desktop\body_model\body_model\Segmentation.nii')  # 替换为你的文件路径
nii_image3 = nib.load(r'C:\Users\allstar\Desktop\a123\ct_213_0000.nii.gz')
body_data = nii_image3.get_fdata()

pixdim = nii_image3.header['pixdim']

# 提取特定切片并转换为灰度图
gray = (body_data[:, :, 57] * 255).astype(np.uint8)

# source_tumour = np.array([186 ,271])
source_tumour = np.array([250 ,90])

targets = np.array([400,130])

# 二值化
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制原始轮廓和近似轮廓
for c in contours:
    # 近似轮廓
    epsilon = 0.005 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)  # 绘制近似轮廓
    
    # 检查每个轮廓上的每条边是否与两点连线相交
    for i in range(len(approx)):
        pt1 = tuple(approx[i][0])
        pt2 = tuple(approx[(i + 1) % len(approx)][0])
        intersection = line_segment_intersection((source_tumour[0], source_tumour[1], targets[0], targets[1]), (pt1[0], pt1[1], pt2[0], pt2[1]))
        if intersection:
            # 交点在轮廓的这条边线段上
            cv2.circle(img, pt1, 5, (0, 255, 0), -1)
            cv2.circle(img, pt2, 5, (0, 255, 0), -1)
            cv2.line(img, tuple(pt1), tuple(pt2), (255, 255, 255), 2)
            
            
            angle = calculate_angle(source_tumour,targets , pt1, pt2)
            print(f"The angle between the two lines is: {angle:.2f} degrees") #[20,160]
            # print(f'交点：{intersection}')

# 绘制两点之间的线段
cv2.line(img, tuple(source_tumour), tuple(targets), (255, 0, 0), 2)

# 绘制点
cv2.circle(img, tuple(source_tumour), 5, (255, 0, 255), -1)
cv2.circle(img, tuple(targets), 5, (255, 255, 0), -1)

# 显示图像
cv2.imshow('Irregular Shape with Approximated Contour', img)
cv2.waitKey(0)
cv2.destroyAllWindows()