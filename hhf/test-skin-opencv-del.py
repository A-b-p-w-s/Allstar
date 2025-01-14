"""
测试opencv分割CT图像中人体皮肤的代码

"""

# import pydicom
# import cv2
# import numpy as np

# # windowing 函数
# def windowing(img, window_width, window_center):
#     minwindow = float(window_center) - 0.5 * float(window_width)
#     new_img = (img - minwindow) / float(window_width)
#     new_img[new_img < 0] = 0
#     new_img[new_img > 1] = 1
#     return (new_img * 255).astype('uint8')

# # 读取DICOM文件的路径
# path = r'C:\Users\Administrator\Desktop\test\skin\slice_0000.dcm'

# # 读取DICOM文件
# dicom_image = pydicom.dcmread(path)

# # 将DICOM图像转换为NumPy数组
# image_array = dicom_image.pixel_array

# # 应用窗宽和窗位
# image_array = windowing(image_array, 500, 0)

# # 上下左右扩充5个像素点，填充值为0
# padded_image = np.pad(image_array, pad_width=5, mode='constant', constant_values=0)

# # 使用Canny算法进行边缘检测
# edges = cv2.Canny(padded_image, threshold1=50, threshold2=150)

# # 寻找边缘图像中的轮廓
# contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# # 如果存在轮廓，计算每个轮廓的面积，并找到面积最大的轮廓
# if contours:
#     contour_areas = [cv2.contourArea(contour) for contour in contours]
#     largest_index = np.argmax(contour_areas)
#     largest_contour = contours[largest_index]

#     # 创建一个与原图像大小相同的掩模
#     mask = np.zeros_like(edges)

#     # 在掩模上绘制最大的轮廓边界
#     cv2.drawContours(mask, [largest_contour], -1, (255), 1)

#     # 应用掩模到扩充后的图像上，只保留最大的轮廓边界，内部像素值设为0
#     padded_image[mask == 0] = 0

#     # 将扩充的部分去掉，恢复到原始图像大小
#     original_size = image_array.shape
#     cropped_image = padded_image[5:5 + original_size[0], 5:5 + original_size[1]]

#     # 显示结果
#     cv2.imshow('Largest Contour Boundary Only', cropped_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("No contours detected.")
#--------------------------------------------------------------------------------------------------
import pydicom
import cv2
import numpy as np

# windowing 函数
def windowing(img, window_width, window_center):
    minwindow = float(window_center) - 0.5 * float(window_width)
    new_img = (img - minwindow) / float(window_width)
    new_img = np.clip(new_img, 0, 1)  # 确保值在0到1之间
    return (new_img * 255).astype('uint8')

# 读取DICOM文件的路径
path = r'C:\Users\Administrator\Desktop\test\skin\slice_0350.dcm'

# 读取DICOM文件
dicom_image = pydicom.dcmread(path)

# 将DICOM图像转换为NumPy数组
image_array = dicom_image.pixel_array

# 应用窗宽和窗位
image_array = windowing(image_array, 500, 0)

# 上下左右扩充5个像素点，填充值为0
padded_image = np.pad(image_array, pad_width=5, mode='constant', constant_values=0)

# 使用Canny算法进行边缘检测
edges = cv2.Canny(padded_image, threshold1=50, threshold2=150)

# 寻找边缘图像中的轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 如果存在轮廓，计算每个轮廓的面积，并找到面积最大的轮廓
if contours:
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    largest_index = np.argmax(contour_areas)
    largest_contour = contours[largest_index]

    # 创建一个与原图像大小相同的掩模
    mask = np.zeros_like(edges)
    # 应用掩模到原始图像上，只保留最大的轮廓宽度为5个像素的部分
    result_image = np.zeros_like(image_array)
    for c in contours:
        if cv2.contourArea(c) == contour_areas[largest_index]:
            cv2.fillPoly(result_image, [c], 255)


    # image = result_image
    # # 使用Canny边缘检测算法
    # edges = cv2.Canny(image, 100, 200)
    # # 创建一个空白图像，用于绘制宽度为5个像素点的边缘
    # height, width = edges.shape
    # result = np.zeros((height, width), dtype=np.uint8)
    # # 遍历边缘图像的每个像素点
    # for y in range(height):
    #     for x in range(width):
    #         if edges[y, x] > 0:
    #             # 如果当前像素点是边缘，将其绘制到结果图像上，宽度为5个像素点
    #             result[max(0, y-1):min(height, y+2), max(0, x-1):min(width, x+2)] = 255


    
    # 显示结果
    cv2.imshow('Largest Contour with 5px Width', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours detected.")
#----------------------------------------------------------------------------------------------------------------------------------------------


# path = r'C:\Users\Administrator\Desktop\test\skin\slice_0350.dcm'

# def windowing(img,window_width,window_center):
#     minwindow = float(window_center)-0.5*float(window_width)
#     new_img = (img-minwindow)/float(window_width)
#     new_img[new_img<0] = 0
#     new_img[new_img>1] = 1
#     return (new_img*255).astype('uint8')


# import pydicom
# import cv2
# import numpy as np

# # 读取DICOM文件
# dicom_image = pydicom.dcmread(path)

# # 将DICOM图像转换为NumPy数组，并转换为8位灰度图像
# image_array = dicom_image.pixel_array

# # binary_image = cv2.threshold(image_array, 175, 255, cv2.THRESH_BINARY)[1]
# # image_array = binary_image

# image_array = windowing(image_array,500,0)


# # 定义一个3x3的结构元素

# # image_array = cv2.GaussianBlur(image_array, (3, 3), 0)
# # image_array = cv2.threshold(image_array, 100, 255, cv2.THRESH_BINARY)[1]
# # image_array = cv2.GaussianBlur(image_array, (3, 3), 0)

# # 应用形态学膨胀操作
# # image_array = cv2.dilate(image_array, kernel, iterations=1)
# # 应用形态学腐蚀操作
# # image_array = cv2.erode(image_array, np.ones((2, 2), np.uint8), iterations=1)

# # image_array = cv2.dilate(image_array, np.ones((3, 3), np.uint8), iterations=1)


# # image_array = cv2.threshold(image_array, 100, 255, cv2.THRESH_BINARY)[1]



# # image_array = np.stack((image_array,) * 3, axis=-1)
# # 上下左右扩充5个像素点，填充值为0
# padded_image = np.pad(image_array, pad_width=5, mode='constant', constant_values=0)

# # 使用高斯模糊减少图像噪声
# # blurred = cv2.GaussianBlur(image_array, (3, 3), 0)

# blurred = padded_image

# # 定义一个3x3的结构元素
# # kernel = np.ones((3, 3), np.uint8)

# # 应用形态学膨胀操作
# # blurred = cv2.dilate(blurred, kernel, iterations=1)

# # cv2.imshow('1', blurred)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# # 使用Canny算法进行边缘检测
# edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# # 寻找边缘图像中的轮廓
# contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# # 如果存在轮廓，计算每个轮廓的面积
# if contours:
#     contour_areas = [cv2.contourArea(contour) for contour in contours]
#     # 找到面积最大的轮廓的索引
#     largest_index = np.argmax(contour_areas)
#     largest_contour = contours[largest_index]

#     # 创建一个与原图像大小相同的掩模
#     mask = np.zeros_like(image_array)

#     # 在掩模上绘制最大的轮廓
#     cv2.drawContours(mask, [largest_contour], -1, (255), -1)

#     # 更新原始图像，只保留最大的轮廓
#     image_array[mask == 0] = 0



# original_size = image_array.shape
# cropped_image = padded_image[5:5+original_size[0], 5:5+original_size[1]]
# # 显示结果
# cv2.imshow('Largest Contour Only', cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# import pydicom
# import cv2
# import numpy as np

# # 读取DICOM文件
# dicom_image = pydicom.dcmread(path)

# # 将DICOM图像转换为NumPy数组
# # DICOM图像通常是16位的，我们需要将其转换为8位以满足OpenCV的要求
# image_array = dicom_image.pixel_array.astype(np.uint8)

# # 转换为灰度图像，DICOM图像已经是灰度的，所以这里只是转换数据类型
# # gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
# gray = image_array

# # 使用高斯模糊减少图像噪声
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # 使用Canny算法进行边缘检测
# edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# # 寻找边缘图像中的轮廓，只检索最外层轮廓
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 计算每个轮廓的面积
# contour_areas = [cv2.contourArea(contour) for contour in contours]

# # 根据面积对轮廓进行降序排序，并获取面积最大的前5个轮廓的索引
# largest_indices = np.argsort(contour_areas)[::-1][:1]

# # 绘制最大的5个闭合边缘
# for index in largest_indices:
#     contour = contours[index]
#     area = contour_areas[index]
#     print(f"Area of the contour: {area}")
#     cv2.drawContours(image_array, [contour], -1, (0, 255, 0), 2)  # 绿色

# # 显示结果
# cv2.imshow('Largest Closed Edges in DICOM Image', image_array)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#--------------------------------------------------------------------------------------------------------------------------------------------------------------

# import cv2
# import numpy as np

# # 读取图像
# image = cv2.imread(path)

# # 转换为灰度图像
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 使用高斯模糊减少图像噪声
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # 使用Canny算法进行边缘检测
# edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# # 寻找边缘图像中的轮廓，只检索最外层轮廓
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 计算每个轮廓的面积
# contour_areas = [cv2.contourArea(contour) for contour in contours]

# # 根据面积对轮廓进行降序排序，并获取面积最大的前5个轮廓的索引
# largest_indices = np.argsort(contour_areas)[::-1][:1]

# # 绘制最大的5个闭合边缘
# for index in largest_indices:
#     contour = contours[index]
#     area = contour_areas[index]
#     print(f"Area of the contour: {area}")
#     cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # 绿色

# # 显示结果
# cv2.imshow('Largest Closed Edges', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()







# path = r'C:\Users\Administrator\Desktop\test\skin\slice_0350.dcm'

# def windowing(img,window_width,window_center):
#     minwindow = float(window_center)-0.5*float(window_width)
#     new_img = (img-minwindow)/float(window_width)
#     new_img[new_img<0] = 0
#     new_img[new_img>1] = 1
#     return (new_img*255).astype('uint8')


# import pydicom
# import cv2
# import numpy as np

# # 读取DICOM文件
# dicom_image = pydicom.dcmread(path)

# # 将DICOM图像转换为NumPy数组，并转换为8位灰度图像
# image_array = dicom_image.pixel_array


# image_array = windowing(image_array,500,0)



# # 上下左右扩充5个像素点，填充值为0
# blurred = np.pad(image_array, pad_width=5, mode='constant', constant_values=0)



# # 使用Canny算法进行边缘检测
# edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# # 寻找边缘图像中的轮廓
# contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# # 如果存在轮廓，计算每个轮廓的面积
# if contours:
#     contour_areas = [cv2.contourArea(contour) for contour in contours]
#     # 找到面积最大的轮廓的索引
#     largest_index = np.argmax(contour_areas)
#     largest_contour = contours[largest_index]

#     # 创建一个与原图像大小相同的掩模
#     mask = np.zeros_like(image_array)

#     # 在掩模上绘制最大的轮廓
#     cv2.drawContours(mask, [largest_contour], -1, (255), -1)

#     # 更新原始图像，只保留最大的轮廓
#     image_array[mask == 0] = 0



# original_size = image_array.shape
# cropped_image = padded_image[5:5+original_size[0], 5:5+original_size[1]]
# # 显示结果
# cv2.imshow('Largest Contour Only', cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()