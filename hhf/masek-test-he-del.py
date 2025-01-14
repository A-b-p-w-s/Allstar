"""
测试将血管标签预处理的代码

"""
import nibabel as nib
import cv2
import numpy as np
from tqdm import tqdm
import os



def a(input_nii_file,input_mask_nii_file,output_file):
    
    
    nifti_entries = os.listdir(input_nii_file)
    nifti_dir = [file for file in nifti_entries if file.endswith('.nii')]
    
    
    mask_entries = os.listdir(input_mask_nii_file)
    mask_dir = [file for file in mask_entries if file.endswith('.nii')]
    

    for dir in tqdm(nifti_dir):
        
        # 打开.nii.gz文件
        nii_file = os.path.join(input_nii_file,dir)
        mask_nii_file = os.path.join(input_mask_nii_file,dir.replace("volume","segmentation"))
        img = nib.load(nii_file)
        mask_img = nib.load(mask_nii_file)

        # 获取数据
        data = img.get_fdata()
        mask = mask_img.get_fdata()

        mask[mask>0]=1

        # 假设数据是3D的，我们有一个轴是切片轴
        slice_axis = 2  # 这取决于你的数据，可能需要调整

        processed_slices=[]


        # 显示每个切片
        for i in range(data.shape[slice_axis]): 
            # 读取图像并转换为灰度图
            image = data[:, :, i]
            
            mask_image = mask[:,:,i]
            
            # 创建结构元素，这里使用5x5的矩形结构
            kernel = np.ones((5, 5), np.uint8)

            aaa=np.array(image * mask_image)
            max_value=np.max(aaa)
            min_value=np.min(aaa)
            
            # 执行膨胀操作
            dilated_image = cv2.dilate(mask_image, kernel, iterations=5)
            dilated_image[dilated_image>0]=1
            
            
            image = image * (dilated_image)
            image[image>max_value]=max_value
            image[image<min_value]=min_value
            
            # 将处理后的切片添加到列表中
            processed_slices.append(image)

        # 将处理后的切片堆叠回3D数组
        processed_data = np.stack(processed_slices, axis=slice_axis)

        # 确保数据类型与原始数据类型相同
        processed_data = processed_data.astype(data.dtype)

        # 创建一个新的Nifti1Image对象
        new_img = nib.Nifti1Image(processed_data, img.affine, img.header)

        # 保存处理后的图像
        save_file = os.path.join(output_file,dir.replace("volume-","Tumour_")+'.gz')
        nib.save(new_img, save_file)
        
        
if __name__ == "__main__":
    
    nii_file = r'C:\Users\allstar\Desktop\images\image'
    mask_nii_file = r'C:\Users\allstar\Desktop\images\label'
    output_file=r'C:\Users\allstar\Desktop\images\tumour-\imagse'

    a(nii_file, mask_nii_file,output_file)
