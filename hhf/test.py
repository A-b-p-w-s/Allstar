import SimpleITK as sitk
import numpy as np

def process_and_save_nifti(nifti_file, output_file):
    # 读取 NIfTI 文件
    image = sitk.ReadImage(nifti_file)
    
    # 将 SimpleITK 图像转换为 NumPy 数组，方便处理
    image_array = sitk.GetArrayFromImage(image)
    
    print(np.unique(image_array))

if __name__ == "__main__":
    # 指定输入和输出文件路径
    nifti_file = r'C:\Users\Administrator\Desktop\111\labelsTr\liver_0.nii.gz'  # 替换为你的 NIfTI 文件路径
    output_file = r'C:\Users\Administrator\Desktop\111\labelsTr\test\liver_0.nii.gz'  # 替换为你想要保存的文件路径
    nifti_file = r'C:\Users\Administrator\Desktop\111\labelsTr\test\liver_0.nii.gz'
    # 调用函数进行处理和保存
    process_and_save_nifti(nifti_file, output_file)