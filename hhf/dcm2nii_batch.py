"""
批量dcm数据转换成png的代码    
用于批量将dcm数据转换成png数据
"""
import SimpleITK as sitk
import os
from tqdm import tqdm

def dcm2nii(dcm_file, nifti_output_dir):
    i = 1
    # 指定目录路径
    directory_path = dcm_file

    # 使用os.listdir()获取目录下的所有文件和文件夹
    entries = os.listdir(directory_path)

    # 过滤出文件夹
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory_path, entry))]
    for dir in tqdm(folders):
        # 设置DICOM文件所在的文件夹路径
        dicom_folder = os.path.join(directory_path, dir)
        
        # 获取DICOM系列的系列ID列表
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_folder)
        
        # 遍历每个系列ID
        for series_id in series_ids:
            # 获取当前系列的文件名列表
            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_folder, series_id)
            
            # 创建ImageSeriesReader对象
            series_reader = sitk.ImageSeriesReader()
            series_reader.SetFileNames(series_file_names)
            
            # 读取DICOM序列并获取3D图像
            image3D = series_reader.Execute()
            
            # 设置输出文件的路径，包含系列ID
            output_file = os.path.join(nifti_output_dir, f"CT_{i:03d}_{series_id}_0000.nii.gz")
            
            # 写入NIfTI格式
            sitk.WriteImage(image3D, output_file)
            
            # 更新系列计数器
        i += 1

if __name__ == "__main__":
    dcm_file = r'C:\Users\allstar\Desktop\dataset\new_CT\new_CT\a' #文件夹
    nifti_output_dir = r'C:\Users\allstar\Desktop\dataset\new_CT\new_CT'
    
    # 确保输出目录存在
    if not os.path.exists(nifti_output_dir):
        os.makedirs(nifti_output_dir)
    
    dcm2nii(dcm_file, nifti_output_dir)

    print("Conversion completed!")



# import SimpleITK as sitk
# import os
# from tqdm import tqdm

# def convert_to_dicom(image3D, output_dir, file_prefix, series_id):
#     # 创建一个ImageFileWriter对象
#     writer = sitk.ImageFileWriter()
    
#     # 设置使用DICOM的标签
#     writer.KeepOriginalMetadata(1)
    
#     # 遍历3D图像中的每个2D层
#     for i, slice in enumerate(image3D):
#         # 为每个2D层创建一个DICOM图像
#         dicom_image = sitk.NaryAdd(image3D, sitk.Cast(slice, sitk.sitkInt16))
        
#         # 设置DICOM图像的元数据
#         dicom_image.SetMetaData('0020|0013', f'{series_id}')
#         dicom_image.SetMetaData('0008|0008', 'DICOM does not have a sequence identifier')
        
#         # 设置输出文件名
#         output_file = os.path.join(output_dir, f"{file_prefix}_{i:04d}.dcm")
        
#         # 写入DICOM格式
#         writer.Execute(dicom_image, output_file)

# def dcm2dcm(dicom_input_dir, dicom_output_dir):
#     i = 1
#     # 过滤出文件夹
#     folders = [entry for entry in os.listdir(dicom_input_dir) if os.path.isdir(os.path.join(dicom_input_dir, entry))]
#     for dir in tqdm(folders):
#         dicom_folder = os.path.join(dicom_input_dir, dir)
#         series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_folder)
#         for series_id in series_ids:
#             series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_folder)
#             series_reader = sitk.ImageSeriesReader()
#             series_reader.SetFileNames(series_file_names)
#             image3D = series_reader.Execute()
#             convert_to_dicom(image3D, dicom_output_dir, f"ct_{i:03d}", series_id)
#         i += 1

# if __name__ == "__main__":
#     dicom_input_dir = r'C:\Users\allstar\Desktop\a'
#     dicom_output_dir = r'C:\Users\allstar\Desktop\ccc'
    
#     # 确保输出目录存在
#     if not os.path.exists(dicom_output_dir):
#         os.makedirs(dicom_output_dir)
    
#     dcm2dcm(dicom_input_dir, dicom_output_dir)
#     print("Conversion completed!")