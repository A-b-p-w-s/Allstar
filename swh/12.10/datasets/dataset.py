import os
from re import escape
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pydicom
from tqdm import tqdm
from torchvision import transforms

class CT_CTA_Dataset(Dataset):
    def __init__(self, ct_dir=None, cta_dir=None, transform=None, args=None):
        """
        初始化CT和CTA图像的Dataset
        :param ct_dir: 存储CT图像文件的文件夹路径
        :param cta_dir: 存储CTA图像文件的文件夹路径
        :param transform: 可选的变换操作（如标准化、裁剪等）
        """
        self.ct_dir = ct_dir
        self.cta_dir = cta_dir
        self.transform = transform
        self.args = args
        
        # 获取CT和CTA文件夹中所有的文件名
        self.ct_files = [f for f in os.listdir(ct_dir) if f.endswith('.nii.gz')]
        self.cta_files = [f for f in os.listdir(cta_dir) if f.endswith('.nii.gz')]
        if self.ct_dir and self.cta_dir:
            assert len(self.ct_files) == len(self.cta_files), "CT和CTA文件数量不匹配"

    def __len__(self):
        # 返回数据集的大小
        return len(self.ct_files)

    def __getitem__(self, idx):
        """
        加载给定索引的CT图像和CTA图像
        :param idx: 数据集中的索引
        :return: 返回CT图像和CTA图像
        """
        # 加载CT图像
        if self.ct_dir:
            ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
            ct_data = self.load_nifti(ct_path)
        
        # 加载CTA图像
        if self.cta_dir:
            cta_path = os.path.join(self.cta_dir, self.cta_files[idx])
            cta_data = self.load_nifti(cta_path)


        if self.ct_dir and self.cta_dir:
            assert ct_data['img'].shape == cta_data['img'].shape, "CT和CTA图像尺寸不匹配"
            if self.transform:
                # ct_data['img'] = self.transform(ct_data['img'], self.args.wc, self.args.ww)
                ct_data['img'] = self.transform(ct_data['img'])
                # ct_data['img'] = norm(ct_data['img'])
                # cta_data['img'] = self.transform(cta_data['img'], self.args.wc, self.args.ww)
                cta_data['img'] = self.transform(cta_data['img'])
                # cta_data['img'] = norm(cta_data['img'])
            return ct_data, cta_data
        elif self.ct_dir:
            if self.transform:
                ct_data['img'] = self.transform(ct_data['img'], self.args.wc, self.args.ww)
                # ct_data['img'] = norm(ct_data['img'])
                # ct_data['img'] = self.transform(ct_data['img'])
            return ct_data
        elif self.cta_dir:
            if self.transform:
                # cta_data['img'] = self.transform(cta_data['img'])
                cta_data['img'] = self.transform(cta_data['img'], self.args.wc, self.args.ww)
                # cta_data['img'] = norm(cta_data['img'])
            return cta_data
        else:
            assert "No CT or CTA files provided"

    def load_nifti(self, file_path):
        """
        从NIfTI文件加载图像数据
        :param file_path: NIfTI文件路径
        :return: PyTorch张量格式的图像数据
        """
        # 使用nibabel加载NIfTI文件
        nii_data = nib.load(file_path)
        affine=nii_data.affine
        header=nii_data.header 
        img_data = nii_data.get_fdata()
        img_data = torch.tensor(img_data, dtype=torch.float32) # [H,W,D]
        
        if img_data.ndimension() == 3:
            img_data = img_data.unsqueeze(0).permute(3,0,1,2)  # 4D: [C, D, H, W]形式
        elif img_data.ndimension() == 2:
            img_data = img_data.unsqueeze(0).unsqueeze(0).permute(3,0,1,2)  # 4D: [C, D, H, W]形式
        
        return {'img': img_data, 'affine':affine, 
                'header':header, 'path':file_path,}
                # 'min':img_data.min(), 'max':img_data.max()}

class DICOM_Dataset(Dataset):
    def __init__(self, ct_dir=None, cta_dir=None, transform=None, args=None):
        """
        初始化CT和CTA图像的Dataset
        :param ct_dir: 存储CT图像文件的文件夹路径
        :param cta_dir: 存储CTA图像文件的文件夹路径
        :param transform: 可选的变换操作（如标准化、裁剪等）
        """
        self.ct_dir = ct_dir
        self.cta_dir = cta_dir
        self.transform = transform
        self.args = args
        
        # 获取CT和CTA文件夹中所有的文件名
        self.ct_files = [f for f in os.listdir(ct_dir) if f.endswith('.dcm')]
        self.cta_files = [f for f in os.listdir(cta_dir) if f.endswith('.dcm')]
        if self.ct_dir and self.cta_dir:
            assert len(self.ct_files) == len(self.cta_files), "CT和CTA文件数量不匹配"
        self.ct_dicom = []
        self.cta_dicom = []

        for dir in self.ct_files:
            for f in os.listdir(os.path.join(ct_dir,dir)):
                self.ct_dicom.append(os.path.join(ct_dir,dir,f))
        for dir in self.cta_files:
            for f in os.listdir(os.path.join(cta_dir,dir)):
                self.cta_dicom.append(os.path.join(cta_dir,dir,f))

    def __len__(self):
        # 返回数据集的大小
        return len(self.ct_dicom)

    def __getitem__(self, idx):
        """
        加载给定索引的CT图像和CTA图像
        :param idx: 数据集中的索引
        :return: 返回CT图像和CTA图像
        """
        # 加载CT图像
        if self.ct_dir:
            ct_data = self.load_dicom(self.ct_dicom[idx])
        
        # 加载CTA图像
        if self.cta_dir:
            cta_data = self.load_dicom(self.cta_dicom[idx])


        if self.ct_dir and self.cta_dir:
            if self.transform:
                ct_data['img'] = self.transform(ct_data['img'], self.args.wc, self.args.ww)
                # ct_data['img'] = norm(ct_data['img'])
                cta_data['img'] = self.transform(cta_data['img'])
                # cta_data['img'] = norm(cta_data['img'])
            return ct_data, cta_data
        elif self.ct_dir:
            if self.transform:
                ct_data['img'] = self.transform(ct_data['img'])
                # ct_data['img'] = norm(ct_data['img'])
            return ct_data
        elif self.cta_dir:
            if self.transform:
                cta_data['img'] = self.transform(cta_data['img'])
                # cta_data['img'] = norm(cta_data['img'])
            return cta_data
        else:
            assert "No CT or CTA files provided"

    def load_dicom(self, file_path):
        """
        从dicom文件加载图像数据
        :param file_path: dicom文件路径
        :return: PyTorch张量格式的图像数据
        """
        ds = pydicom.dcmread(file_path)
    
        # 获取DICOM文件中的图像数据
        img_data = ds.pixel_array
        
        img_data = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0) # [1,H,W]
        dir_name = os.path.dirname(file_path)
        # return img_data
        return {'img': img_data, 'path':dir_name}

# norm = transforms.Normalize(mean=[0.485], std=[0.229])

def collate_fn_nii(batch):
    """
    :param batch: 一个包含多个数据点元组的列表
    :return: 一个包含批量数据的元组
    """
    if len(batch[0]) == 2:
        ct_data, cta_data = batch[0]
        batch = (ct_data, cta_data)
        return batch
    else:
        item = batch[0]
        return item

def collate_fn_dicom(batch):
    """
    :param batch: 一个包含多个数据元组的列表
    :return: 一个包含批量数据的元组
    """
    if isinstance(batch[0], (tuple)):
        ct_list = []
        cta_list = []   
        for t in batch:
            ct_list.append(t[0])
            cta_list.append(t[1])
        ct_imgs = [item['img'] for item in ct_list]
        ct_paths = [item['path'] for item in ct_list]
        ct_imgs = torch.stack(ct_imgs)
        cta_imgs = [item['img'] for item in cta_list]
        cta_paths = [item['path'] for item in cta_list]
        cta_imgs = torch.stack(cta_imgs)
        return {'ct_imgs': ct_imgs, 'ct_paths': ct_paths,
                'cta_imgs': cta_imgs, 'cta_paths': cta_paths}
    
    elif isinstance(batch[0], (dict)):
        imgs = [item['img'] for item in batch]
        paths = [item['path'] for item in batch]
        imgs = torch.stack(imgs)
        return {'imgs': imgs, 'paths': paths}
    else:
        raise ValueError('invalid datasets type')

def map_data(tensor, min=-1, max=1): 
    """input: tensor, min, max
    tensor: 待转换的tensor
    min: 窗位 - 窗宽/2 or norm -1 or liver: -115
    max: 窗位 + 窗宽/2 or norm 1  or liver: 185
    """
    # 获取tensor的最小值和最大值
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    # 按照线性映射公式进行转换
    mapped_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (max - min) + min
    
    return mapped_tensor

def map_data_2(image, WC=30, WW=400):
    
    center = WC # 40 400//60 300
    width = WW # 200
    try:
        win_min = center - width / 2.0 + 0.5
        win_max = center + width / 2.0 + 0.5
    except:
        # print(WC[0])
        # print(WW[0])
        center = WC[0]  # 40 400//60 300
        width = WW[0]  # 200
        win_min = center - width / 2.0 + 0.5
        win_max = center + width / 2.0 + 0.5

    # image = torch.trunc(image)
    image[image > win_max] = win_max
    image[image < win_min] = win_min
    image = image / float(width) 
    # 获取tensor的最小值和最大值
    tensor_min = image.min()
    tensor_max = image.max()

    # 按照线性映射公式进行转换
    image = 2 * (image - tensor_min) / (tensor_max - tensor_min) - 1

    return image

def map_data_3(image, WC=30, WW=300):
    
    center = WC # 40 400//60 300
    width = WW # 200
    try:
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5
    except:
        # print(WC[0])
        # print(WW[0])
        center = WC[0]  # 40 400//60 300
        width = WW[0]  # 200
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5
    dFactor = 255.0 / (win_max - win_min)
    image = image - win_min
    image = torch.trunc(image * dFactor)
    image[image > 255] = 255
    image[image < 0] = 0
    image = image / 255.0 
    image = (image - 0.5) * 2
    return image

def unmap_data_3(image, WC=30, WW=300):
    # 首先，我们需要将图像的值从映射后的值转换回原始的映射值
    image = (image / 2.0) + 0.5  # 反做映射后的乘2和减0.5操作
    image = image * 255.0  # 反做归一化操作
    image = torch.round(image)  # 将浮点数四舍五入到最近的整数
    image[image > 255] = 255
    image[image < 0] = 0
    center = WC  
    width = WW  
    win_min = (2 * center - width) / 2.0 + 0.5
    win_max = (2 * center + width) / 2.0 + 0.5
    dFactor = 255.0 / (win_max - win_min)
    image = image / dFactor  # 反做缩放操作
    image = image + win_min  # 反做平移操作
    return image

def unmap_data_2(tensor, wc=30, ww=400):
    # 反映射公式
    reversed_tensor = (tensor + 1) / 2.0 * ww - 170
    return reversed_tensor


if __name__ == '__main__':
    ct_files = r'D:\data\DICOM_CTA\train\A3' #平衡期 类似平扫
    cta_files_A0 = r'D:\data\DICOM_CTA\train\A0' #动脉期
    cta_files_A1 = r'D:\data\DICOM_CTA\train\A1' #静脉前期
    cta_files_A2 = r'D:\data\DICOM_CTA\train\A2' #静脉后期
    # transform=transforms.Compose(
    # transforms.ToTensor())
    dataset = DICOM_Dataset(cta_files_A0) #     
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_dicom) # here batch_size must should be 1
    for ct_data in tqdm(dataloader):
        if  torch.isnan(ct_data["imgs"].max()):
            print(ct_data["paths"])
        pass
    
