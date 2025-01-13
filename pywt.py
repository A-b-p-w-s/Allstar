import os
import pydicom
import numpy as np
import pywt
import matplotlib.pyplot as plt

# 加载 DICOM 图像
def load_dicom(file_path):
    ds = pydicom.dcmread(file_path)
    img = ds.pixel_array.astype(np.float32)
    return img, ds  # 返回像素数据和 DICOM 对象以便保存

# 小波去噪
def wavelet_denoise(img, wavelet='db1', level=2, threshold=20):
    # 小波分解
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    # 阈值处理
    coeffs_denoised = []
    for coeff in coeffs:
        if isinstance(coeff, tuple):
            coeffs_denoised.append(tuple(pywt.threshold(c, threshold, mode='soft') for c in coeff))
        else:
            coeffs_denoised.append(coeff)
    # 小波重构
    img_denoised = pywt.waverec2(coeffs_denoised, wavelet=wavelet)
    return img_denoised

# 处理文件夹中的所有 DICOM 图像
def process_dicom_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path) and file_name.lower().endswith('.dcm'):
            print(f"Processing: {file_name}")
            
            # 加载 DICOM 图像
            img, ds = load_dicom(file_path)
            
            # 小波去噪
            denoised_img = wavelet_denoise(img)
            
            # 保存处理后的图像
            save_dicom(denoised_img, ds, os.path.join(output_folder, file_name))

# 保存处理后的 DICOM 图像
def save_dicom(img, ds, output_path):
    img = np.clip(img, 0, np.max(img)).astype(ds.pixel_array.dtype)  # 保留原始数据类型
    ds.PixelData = img.tobytes()
    ds.save_as(output_path)
    print(f"Saved: {output_path}")


images_root = r'C:\Users\Administrator\Desktop\ves\imagesTr_dicom'
output_folder = r'C:\Users\Administrator\Desktop\ves\processed_images'
process_dicom_folder(images_root, output_folder)
