import nibabel as nib

# 加载.nii.gz文件
nii_img = nib.load(r'C:\Users\Administrator\Desktop\test\volume-325.nii')  # 替换为你的文件路径
data = nii_img.get_fdata()

# 假设骨头的HU值范围是200到1000
bone_hu_min = 300
bone_hu_max = 1000

# 将非骨头部分的像素值设为0
data[data < bone_hu_min] = 0
data[data > bone_hu_max] = 0


# 可选：保存修改后的图像
new_nii_img = nib.Nifti1Image(data, nii_img.affine, nii_img.header)
nib.save(new_nii_img, r'C:\Users\Administrator\Desktop\test\volume-325-bone.nii')  # 替换为保存路径

print("Conversion completed")