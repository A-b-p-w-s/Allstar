import os
import pydicom
import numpy as np
import pywt

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

# 递归处理文件夹中的所有 DICOM 图像
def process_dicom_folder_recursive(folder_path, output_folder):
    for root, _, files in os.walk(folder_path):
        relative_path = os.path.relpath(root, folder_path)  # 计算相对路径
        output_subfolder = os.path.join(output_folder, relative_path)  # 构造对应的输出文件夹路径
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)  # 创建输出子文件夹

        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.lower().endswith('.dcm'):  # 仅处理 .dcm 文件
                print(f"Processing: {file_path}")

                # 加载 DICOM 图像
                img, ds = load_dicom(file_path)

                # 小波去噪
                denoised_img = wavelet_denoise(img)

                # 保存处理后的图像
                output_file_path = os.path.join(output_subfolder, file_name)
                save_dicom(denoised_img, ds, output_file_path)

# 保存处理后的 DICOM 图像
def save_dicom(img, ds, output_path):
    img = np.clip(img, 0, np.max(img)).astype(ds.pixel_array.dtype)  # 保留原始数据类型
    ds.PixelData = img.tobytes()
    ds.save_as(output_path)
    print(f"Saved: {output_path}")

# 设置路径
images_root = r'C:\Users\Administrator\Desktop\ves\imagesTr_dicom'
output_folder = r'C:\Users\Administrator\Desktop\ves\processed_images'

# 开始处理
process_dicom_folder_recursive(images_root, output_folder)
