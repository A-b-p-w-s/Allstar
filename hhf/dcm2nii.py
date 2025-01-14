"""
dcm数据转换成nii的代码    
用于将dcm数据转换成nii数据
"""
import SimpleITK as sitk
import os
from tqdm import tqdm

def dcm2nii(dcm_file, nifti_output_dir):

    i=1
    # 指定目录路径
    directory_path = dcm_file

    # 使用os.listdir()获取目录下的所有文件和文件夹
    entries = os.listdir(directory_path)

    # 过滤出文件夹
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory_path, entry))]
    for dir in tqdm(folders):
        
        # 使用os.makedirs()创建文件夹
        try:
            os.makedirs(os.path.join(nifti_output_dir,dir))
            print(f"文件夹 '{os.path.join(nifti_output_dir,dir)}' 创建成功。")
        except FileExistsError:
            print(f"文件夹 '{os.path.join(nifti_output_dir,dir)}' 已存在。")
        except Exception as e:
            print(f"创建文件夹时出错：{e}")
            
        # 设置DICOM文件所在的文件夹路径
        dicom_folder = os.path.join(dcm_file,dir)
        # 设置输出文件的路径
        output_file = os.path.join(nifti_output_dir,dir,f"ct_213_0000.nii.gz")

        # 获取DICOM系列的文件名列表
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_folder)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_folder)

        # 创建ImageSeriesReader对象
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)

        # 读取DICOM序列并获取3D图像
        image3D = series_reader.Execute()

        # 写入NIfTI格式
        sitk.WriteImage(image3D, output_file)
        i = i+1


if __name__ == "__main__":
    dcm_file = r'C:\Users\allstar\Desktop\213\a'
    nifti_output_dir = r'C:\Users\allstar\Desktop\aaaaa'
    
    dcm2nii(dcm_file, nifti_output_dir)

    print("Conversion completed!")