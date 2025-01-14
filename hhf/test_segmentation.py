"""
用于分割mask的nii数据中多标签去除不要标签的代码

"""
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm

def a(nifti_file,dicom_output_dir):

    if not os.path.exists(dicom_output_dir):
        os.makedirs(dicom_output_dir)

    entries = os.listdir(nifti_file)
    
    # 筛选出以.nii.gz结尾的文件
    nifti_gz_files = [file for file in entries if file.endswith('.nii.gz')]

    for dir in tqdm(nifti_gz_files):
        input_dir=os.path.join(nifti_file,dir)
        # 加载.nii.gz文件
        nii_img = nib.load(input_dir)  # 替换为你的文件路径
        data = nii_img.get_fdata()
        print(np.unique(data))
        # # print(np.unique(data))
        # # 分割需要的部分
        # # data[data >0 ] = 255
        # # print('\n',dir)
        # data[data<1.1 ] = 0
        # data[data>0] = 1

        # # data[data >0.1 ] = 1
        # # print(np.unique(data))
        # # 可选：保存修改后的图像
        # new_nii_img = nib.Nifti1Image(data, nii_img.affine, nii_img.header)

        # save_dir=os.path.join(dicom_output_dir,dir)
        # nib.save(new_nii_img, save_dir)  # 替换为保存路径

if __name__ == "__main__":
    nifti_file = r'D:\new30data\肝脏分割\肝脏seg'
    dicom_output_dir = r'D:\data\Task03_Liver\Task03_Liver\a'
    a(nifti_file, dicom_output_dir)
    print("Conversion completed")