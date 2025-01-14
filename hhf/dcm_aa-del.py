"""
测试找到dcm数据的真实xyz坐标和ct上xyz坐标的对应关系的代码
"""
import nibabel as nib
import numpy as np
import glob
import os
import pydicom
import cv2

def read_dicom_series(directory):
    # 获取文件夹中所有.dcm文件的路径
    dcm_files = sorted(glob.glob(os.path.join(directory, '*.dcm'))) 
    # 读取第一个DICOM文件以获取图像尺寸和深度信息
    ref_image = pydicom.dcmread(dcm_files[0])

    pixel_spacing = ref_image.PixelSpacing # 返回一个包含x和y距离的元组

    # 获取体素的z方向距离（层间距）
    slice_thickness = ref_image.SliceThickness # 返回z方向的距离
    pixdim = [None] * 4
    pixdim[1]=pixel_spacing[0]
    pixdim[2]=pixel_spacing[1]
    pixdim[3]=slice_thickness
    
    
    # 初始化一个3D数组来存储所有DICOM图像
    images = np.stack([pydicom.dcmread(f).pixel_array for f in dcm_files])
    
    return np.transpose(images, (2, 1, 0)),pixdim

def optimization_points(nii_data):
    array = []   
    
    for i in range(nii_data.shape[2]):
        array.append(nii_data[:, :, i])
    array = np.array(array)
    array = array.transpose((1, 2, 0))
    
    print(array.shape)
        
 
    array2 = []   
    
    for i in range(nii_data.shape[0]):
        array2.append(nii_data[i, :, :])
    array2 = np.array(array2)

    print(array2.shape)
    # array = array.transpose((1, 2, 0))
    # print(array.shape)
    
    return array

nii_image3 = nib.load(r'C:\Users\allstar\Desktop\abc\213\liver-213\CT_003_1.2.3.4.5_0000.nii.gz')
body_data1 = nii_image3.get_fdata()


body_data2,_ = read_dicom_series(r"C:\Users\allstar\Desktop\abc\213-new\liver-213")
body_data2 = optimization_points(body_data2)

print(np.array_equal(body_data1,body_data2))