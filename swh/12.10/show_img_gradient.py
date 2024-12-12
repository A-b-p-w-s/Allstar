from criterion import compute_edge
from datasets.dataset import collate_fn_nii, map_data_2, map_data_3, CT_CTA_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    ct_files = r'E:\data\CTA\test\A3' #平衡期 类似平扫
    cta_files_A0 = r'E:\data\CTA\test\A0' #动脉期
    cta_files_A1 = r'E:\data\CTA\test\A1' #静脉前期
    cta_files_A2 = r'E:\data\CTA\test\A2' #静脉后期
    # transform=transforms.Compose(
    # transforms.ToTensor())
    dataset = CT_CTA_Dataset(ct_files, cta_files_A0, transform=map_data_2) #     
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_nii) # here batch_size must should be 1
    for ct_data, cta_data in tqdm(dataloader):
        ct_img = ct_data['img'].squeeze(0)
        cta_img = cta_data['img'].squeeze(0)
        edg1 = compute_edge(ct_img[100,:,:], kernel='sobel')
        edg2 = compute_edge(cta_img[100,:,:], kernel='sobel')
        edg1 = edg1.cpu().numpy().transpose(1, 2, 0)
        edg2 = edg2.cpu().numpy().transpose(1, 2, 0)

        # 创建一个包含两个子图的图表
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
        axes = axes.ravel()
        # 在第一个子图中显示第一张图像
        axes[0].imshow(edg1, cmap='gray')  # 使用 'gray' 颜色映射来正确显示灰度图像
        axes[0].set_title('edg1')
        axes[0].axis('off')  # 关闭坐标轴

        # 在第二个子图中显示第二张图像
        axes[1].imshow(edg2, cmap='gray')
        axes[1].set_title('edg2')
        axes[1].axis('off')  # 关闭坐标轴

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.show()
        break