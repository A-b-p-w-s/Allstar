import SimpleITK as sitk
import os
from tqdm import tqdm

def dcm2nii(dcm_file, nifti_output_dir):
    i = 335
    # 指定目录路径
    directory_path = dcm_file

    # 使用os.listdir()获取目录下的所有文件和文件夹
    entries = os.listdir(directory_path)

    # 过滤出文件夹
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory_path, entry))]
    for dir in folders:
        entries_2 = os.listdir(directory_path+'\\'+dir)
        for dir_2 in entries_2:
    
            print('='*180)
            # 设置DICOM文件所在的文件夹路径
            dicom_folder = os.path.join(directory_path, dir,dir_2,'ST0')
            entries3 = os.listdir(dicom_folder)
            abc={}
            cba={}
            cba_id={}
            for dir3 in entries3:
                # 获取DICOM系列的系列ID列表
                series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(os.path.join(dicom_folder, dir3))
                if len(os.listdir(os.path.join(dicom_folder, dir3))) in abc:
                    abc[len(os.listdir(os.path.join(dicom_folder, dir3)))] = abc[len(os.listdir(os.path.join(dicom_folder, dir3)))]+1
                    cba[len(os.listdir(os.path.join(dicom_folder, dir3)))].append(dir3)
                else:
                    abc[len(os.listdir(os.path.join(dicom_folder, dir3)))]=1
                    cba[len(os.listdir(os.path.join(dicom_folder, dir3)))] = [dir3]
                cba_id[dir3]=series_ids
                print('--------name:',dir3.replace('SE',''),'--------id:',series_ids,'--------num:',len(os.listdir(os.path.join(dicom_folder, dir3))),'--------')
            num_4 = [k for k, v in abc.items() if v == 4]
            for n in num_4:
                i+=1
                # 确保输出目录存在
                if not os.path.exists(os.path.join(nifti_output_dir,'A0')):
                    os.makedirs(os.path.join(nifti_output_dir,'A0'))
                    os.makedirs(os.path.join(nifti_output_dir,'A1'))
                    os.makedirs(os.path.join(nifti_output_dir,'A2'))
                    os.makedirs(os.path.join(nifti_output_dir,'A3'))
                    
                sorted_list = sorted(cba[n], key=lambda s: int(s[2:]))
                save_ct_id=0
                for j in sorted_list:
                    # print(cba_id[j][0])
                    # print(os.path.join(dicom_folder,j))
                    # 获取当前系列的文件名列表
                    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(os.path.join(dicom_folder,j), cba_id[j][0])
                    
                    # 创建ImageSeriesReader对象
                    series_reader = sitk.ImageSeriesReader()
                    series_reader.SetFileNames(series_file_names)
                    
                    # 读取DICOM序列并获取3D图像
                    image3D = series_reader.Execute()
                    
                    # 设置输出文件的路径，包含系列ID
                    output_file = os.path.join(nifti_output_dir,f'A{save_ct_id}',f"CTA_{i:03d}_0000.nii.gz")
                    print(output_file)
                    save_ct_id+=1

                    
                    # 写入NIfTI格式
                    sitk.WriteImage(image3D, output_file)
        

if __name__ == "__main__":
    dcm_file = r'F:\0A-郑老师数据存放文件夹\CT门静脉血管成像'
    nifti_output_dir = r'F:\CTAtest'
    
    
    dcm2nii(dcm_file, nifti_output_dir)

    print("Conversion completed!")
