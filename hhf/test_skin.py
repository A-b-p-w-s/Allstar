"""
用于分割CT图像中人体皮肤的代码

"""
import SimpleITK as sitk
import os
import numpy as np
import cv2
from tqdm import tqdm
import nibabel as nib
import pydicom

# windowing 函数
def windowing(img, window_width, window_center):
    minwindow = float(window_center) - 0.5 * float(window_width)
    new_img = (img - minwindow) / float(window_width)
    new_img[new_img < 0] = 0
    new_img[new_img > 1] = 1
    return (new_img * 255).astype('uint8')

# # 读取DICOM文件的路径
# path = r'C:\Users\Administrator\Desktop\test\skin\slice_0350.dcm'

# # 读取DICOM文件
# dicom_image = pydicom.dcmread(path)

def b(image_array):
    # 应用窗宽和窗位
    image_array = windowing(image_array, 500, 0)
    # image_array = cv2.erode(image_array, np.ones((3, 3), np.uint8), iterations=2)
    # image_array = cv2.dilate(image_array, np.ones((3, 3), np.uint8), iterations=2)
    # image_array = cv2.threshold(image_array, 125, 255, cv2.THRESH_BINARY)[1]

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
        # 使用Canny边缘检测算法
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

        original_size = image_array.shape
        cropped_image = result_image[5:5 + original_size[0], 5:5 + original_size[1]]
        
        return cropped_image



def a(dicom_image):
    # 将DICOM图像转换为NumPy数组
    image_array = dicom_image

    # 应用窗宽和窗位
    image_array = windowing(image_array, 50, 200)
    
    image_array = cv2.erode(image_array, np.ones((3, 3), np.uint8), iterations=5)
    image_array = cv2.dilate(image_array, np.ones((3, 3), np.uint8), iterations=5)
    
    image_array = cv2.GaussianBlur(image_array, (5, 5), 2)
    
    # 上下左右扩充5个像素点，填充值为0
    padded_image = np.pad(image_array, pad_width=5, mode='constant', constant_values=0)

    # 使用Canny算法进行边缘检测
    edges = cv2.Canny(padded_image, threshold1=0, threshold2=100)

    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=5)
    aaa = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=5)
    
    # edges = cv2.Canny(edges, threshold1=0, threshold2=100)



    # 寻找边缘图像中的轮廓
    contours, _ = cv2.findContours(aaa, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 如果存在轮廓，计算每个轮廓的面积，并找到面积最大的轮廓
    if contours:
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        largest_index = np.argmax(contour_areas)
        largest_contour = contours[largest_index]

        # 创建一个与原图像大小相同的掩模
        mask = np.zeros_like(edges)
        
        # 在掩模上绘制最大的轮廓边界
        cv2.drawContours(mask, [largest_contour], -1, (255), 2)

        # 应用掩模到扩充后的图像上，只保留最大的轮廓边界，内部像素值设为0
        # padded_image[mask == 0] = 0

        # # 将扩充的部分去掉，恢复到原始图像大小
        # original_size = image_array.shape
        cropped_image = mask[5:-5, 5:-5]
        
        
        cv2.imshow('edges', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cropped_image[cropped_image>0]  = 255
        return cropped_image




# def convert_nifti_to_dicom(nifti_file, dicom_output_dir):
#     # Read the NIfTI file
#     nifti_image = sitk.ReadImage(nifti_file)
#     sitk.GetArrayFromImage
#     nifti_image = sitk.Cast(nifti_image, sitk.sitkInt16)
#     # Get image size, spacing, and direction
#     size = nifti_image.GetSize()
#     spacing = nifti_image.GetSpacing()
#     direction = nifti_image.GetDirection()
#     origin = nifti_image.GetOrigin()
    
#     # Create the DICOM series writer
#     writer = sitk.ImageFileWriter()
#     writer.KeepOriginalImageUIDOn()
    
#     # Get the number of slices in the NIfTI image
#     num_slices = size[2]
    
#     # Make sure the output directory exists
#     if not os.path.exists(dicom_output_dir):
#         os.makedirs(dicom_output_dir)
    
#     # Define some basic DICOM tags (you can customize these as needed)
#     tags = {
#         "0010|0010": "Test^Patient",        # Patient Name
#         "0020|000D": "1.2.3.4",             # Study Instance UID
#         "0020|000E": "1.2.3.4.5",           # Series Instance UID
#         "0008|0020": "20240101",            # Study Date
#         "0008|0030": "090000",              # Study Time
#         "0008|0050": "123456",              # Accession Number
#         "0008|0060": "MR",                  # Modality
#         "0020|0011": "1",                   # Series Number
#         "0008|103E": "Test Series"          # Series Description
#     }


#     for i in tqdm(range(num_slices)):
#         # Extract the ith slice from the NIfTI image
#         slice_i = sitk.GetArrayFromImage(nifti_image[:, :, i])
        
#         # Process the slice using the 'a' function (which should return a numpy array)
#         processed_slice_array = a(slice_i)
        
#         # Convert the processed numpy array back to a SimpleITK image
#         processed_slice_image = sitk.GetImageFromArray(processed_slice_array)


        
#         # Set the file name for the DICOM file
#         dicom_file = os.path.join(dicom_output_dir, f"slice_{i:04d}.dcm")
        
#         # Set the DICOM tags

#         for tag, value in tags.items():
#             processed_slice_image.SetMetaData(tag, value)
        
#         # Update Instance Number and Image Position for each slice
#         processed_slice_image.SetMetaData("0020|0013", str(i + 1))  # Instance Number
#         processed_slice_image.SetMetaData("0020|0032", "\\".join(map(str, [
#             origin[0],
#             origin[1],
#             origin[2] + i * spacing[2]
#         ])))  # Image Position (Patient)
#         processed_slice_image.SetMetaData("0020|0037", "\\".join(map(str, direction[:6])))  # Image Orientation (Patient)
#         processed_slice_image.SetMetaData("0018|0050", str(spacing[2]))  # Slice Thickness
#         processed_slice_image.SetMetaData("0018|0088", str(spacing[2]))  # Spacing Between Slices
        
#         # print(processed_slice_image)

#         # Write the DICOM file
#         writer.SetFileName(dicom_file)
#         writer.Execute(processed_slice_image)
    
#     print(f"Conversion completed: {dicom_output_dir}")


def convert_nifti_to_dicom(dcm_folder, output_folder):
    # 创建输出文件夹，如果不存在的话
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹中所有 .dcm 文件
    dcm_files = os.listdir(dcm_folder)

    # 读取 DICOM 文件并存储切片信息
    dicom_data = []
    for dcm_file in dcm_files:
        dcm_path = os.path.join(dcm_folder, dcm_file)
        ds = pydicom.dcmread(dcm_path)
        
        # 获取 InstanceNumber 或 SliceLocation
        instance_number = None
        if 'InstanceNumber' in ds:
            instance_number = ds.InstanceNumber
        elif 'SliceLocation' in ds:
            instance_number = ds.SliceLocation

        # 如果切片顺序信息存在，添加到 dicom_data 列表中
        if instance_number is not None:
            dicom_data.append((ds, instance_number))

    # 按切片顺序（InstanceNumber 或 SliceLocation）排序
    dicom_data.sort(key=lambda x: x[1])

    # 读取 PNG 文件并保存为 DICOM 文件
    for i, (ds, _) in enumerate(dicom_data):
    
        # 获取图像的色彩类型和位深（假设是灰度图）
        bits_allocated = ds.BitsAllocated

        pixel_data = ds.pixel_array
        
        pixel_data = a(pixel_data)

        
        
        # 创建一个新的黑色图像（灰度图像，全黑）
        if bits_allocated == 16:
            black_image = np.array(pixel_data).astype(np.uint16)
        else:
            black_image = np.array(pixel_data).astype(np.uint8)
         
        
        # 创建新的 DICOM 数据集
        new_ds = ds.copy()
        # 替换 PixelData为全黑的图像
        new_ds.PixelData = black_image.tobytes()
        
        # 确保其他元数据正确
        new_ds.InstanceNumber = i + 1  # 保证切片的顺序

        # 保存新的黑色切片 DICOM 文件
        output_dcm_path = os.path.join(output_folder, f"black_slice_{i+1}.dcm")
        # new_ds.save_as(output_dcm_path)
        print(f"保存新的 DICOM 文件：{output_dcm_path}")


if __name__ == "__main__":
    
    dicom_folder = r'C:\Users\allstar\Desktop\dataset\new_CT\new_CT\CT'
    dicom_output_dir = r'C:\Users\allstar\Desktop\dataset\new_CT\new_CT\skin_new'

    convert_nifti_to_dicom(dicom_folder, dicom_output_dir)