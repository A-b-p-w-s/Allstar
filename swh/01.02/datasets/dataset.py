import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pydicom
from tqdm import tqdm
import numpy as np

class CT_CTA_Dataset(Dataset):
    def __init__(self, ct_dir=None, cta_dir=None, args=None):
        """
        初始化CT和CTA图像的Dataset
        :param ct_dir: 存储CT图像文件的文件夹路径
        :param cta_dir: 存储CTA图像文件的文件夹路径
        """
        self.ct_dir = ct_dir
        self.cta_dir = cta_dir
        self.args = args
        
        # 获取CT和CTA文件夹中所有的文件名
        if self.ct_dir is not None:
            self.ct_files = [f for f in os.listdir(ct_dir) if f.endswith('.nii.gz')]
        if self.cta_dir is not None:
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

            ct_data['img'] = map_data(ct_data['img'], self.args.wc, self.args.ww)
            cta_data['img'] = map_data(cta_data['img'], self.args.wc, self.args.ww)

            return ct_data, cta_data
        elif self.ct_dir:
            # ct_data['img'] = norm(ct_data['img'])
            ct_data['img'] = map_data(ct_data['img'], self.args.wc, self.args.ww)
            return ct_data
        elif self.cta_dir:
            cta_data['img'] = map_data(cta_data['img'], self.args.wc, self.args.ww)
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
        img_data = nii_data.get_fdata(dtype=np.float32)
        img_data = torch.from_numpy(img_data) # [H,W,D]
        
        if img_data.ndimension() == 3:
            img_data = img_data.unsqueeze(0)  # 4D: [C, D, H, W]形式
        elif img_data.ndimension() == 2:
            img_data = img_data.unsqueeze(0).unsqueeze(0)  # 4D: [C, D, H, W]形式
        
        return {'img': img_data, 'affine':affine, 
                'header':header, 'path':file_path,}
                # 'min':img_data.min(), 'max':img_data.max()}

class DICOM_Dataset(Dataset):
    def __init__(self, ct_dir=None, cta_dir=None, args=None):
        """
        初始化CT和CTA图像的Dataset
        :param ct_dir: 存储CT图像文件的文件夹路径
        :param cta_dir: 存储CTA图像文件的文件夹路径
        """
        self.ct_dir = ct_dir
        self.cta_dir = cta_dir
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
            ct_data['img'] = map_data(ct_data['img'], self.args.wc, self.args.ww)
            # ct_data['img'] = norm(ct_data['img'])
            cta_data['img'] = map_data(cta_data['img'], self.args.wc, self.args.ww)
            # cta_data['img'] = norm(cta_data['img'])
            return ct_data, cta_data
        elif self.ct_dir:
            ct_data['img'] = map_data(ct_data['img'], self.args.wc, self.args.ww)
            # ct_data['img'] = norm(ct_data['img'])
            return ct_data
        elif self.cta_dir:
            cta_data['img'] = map_data(cta_data['img'], self.args.wc, self.args.ww)
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

def map_data(image, WC=30.0, WW=400.0):
    
    win_min = WC - WW / 2.0 
    win_max = WC + WW / 2.0 

    # image = torch.trunc(image)
    image.clamp_(min=win_min, max=win_max)
    image.sub_(win_min)
    image.div_(WW) 

    return image

def unmap_data(img, wc=30.0, ww=400.0):
    # 反映射公式
    win_min = wc - ww / 2.0
    reversed_img = img * ww + win_min
    return reversed_img



if __name__ == '__main__':
    ct_files = r'D:\data\CTA\train\A3' #平衡期 类似平扫
    cta_files_A0 = r'D:\data\CTA\val\A0' #动脉期
    cta_files_A1 = r'D:\data\CTA\train\A1' #静脉前期
    cta_files_A2 = r'D:\data\CTA\train\A2' #静脉后期
    # transform=transforms.Compose(
    # transforms.ToTensor())
    import argparse
    parser = argparse.ArgumentParser("")
    parser.add_argument('--wc', type=float, default=0, help='window center')
    parser.add_argument('--ww', type=float, default=400, help='window width')
    args = parser.parse_args()
    dataset = CT_CTA_Dataset(cta_files_A0, args=args) #     
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_nii) # here batch_size must should be 1
    for ct_data in tqdm(dataloader):
        if  ct_data['img'].shape[0] < 55:
            print(ct_data["path"])
        pass
    
