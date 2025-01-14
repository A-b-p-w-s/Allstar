"""
测试剔除mask的nii数据标签中不要的标签的代码

"""
import SimpleITK as sitk
import numpy as np
import glob
import os

def process_and_save_nifti(nifti_file, output_file,img_ids):

    for i in img_ids:
        label_path=os.path.join(nifti_file,i)
        save_path=os.path.join(output_file,i)
        # print(label_path)
        # print(save_path)

        # 读取 NIfTI 文件
        image = sitk.ReadImage(label_path)
        
        # 将 SimpleITK 图像转换为 NumPy 数组，方便处理
        image_array = sitk.GetArrayFromImage(image)
        
        # 处理每个切片，大于0的设置为1
        processed_array = np.where(image_array > 0, 1, image_array)
        
        # 将处理后的 NumPy 数组转换回 SimpleITK 图像
        processed_image = sitk.GetImageFromArray(processed_array)
        
        # 确保新图像具有与原始图像相同的元数据
        processed_image.CopyInformation(image)
        
        # 保存处理后的图像为新的 NIfTI 文件，使用gzip压缩
        sitk.WriteImage(processed_image, save_path, True)
    print("Conversion completed")
if __name__ == "__main__":
    # 指定输入和输出文件路径
    nifti_file = r'G:\hhf\data\Task03_Liver\labelsTr'  # 替换为你的 NIfTI 文件路径
    img_ids = glob.glob(os.path.join(nifti_file,'*.nii.gz'))
    img_ids = [os.path.splitext(os.path.basename(p))[0]+'.gz' for p in img_ids]
    output_file = r'G:\hhf\data\Task03_Liver\0-1\labelsTr'  # 替换为你想要保存的文件路径

    # 调用函数进行处理和保存
    process_and_save_nifti(nifti_file, output_file,img_ids)