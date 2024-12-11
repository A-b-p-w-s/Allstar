import torch.utils.data as data
import os
import pydicom
import numpy as np
from PIL import Image

class VesselDataset(data.Dataset):
    def __init__(self, images_root, labels_root, transform=None, target_transform=None, hu_window=None):
        """
        :param images_root: 图像文件根目录
        :param labels_root: 标签文件根目录
        """
        self.data_paths = []

        # 遍历 images_root 中的文件夹
        for folder_name in os.listdir(images_root):
            image_folder_path = os.path.join(images_root, folder_name)
            label_folder_path = os.path.join(labels_root, folder_name)
            print(image_folder_path)
            print(label_folder_path)

            # 遍历 image_folder 中的文件
            for image_file in os.listdir(image_folder_path):
                if image_file.endswith('.dcm'):
                    image_path = os.path.join(image_folder_path, image_file)
                    label_path = os.path.join(label_folder_path, image_file)

                    # 确保 label 文件存在
                    if os.path.exists(label_path):
                        self.data_paths.append((image_path, label_path))
                    else:
                        print(f"标签文件缺失: {label_path}")

        print(f"成功加载数据对数量: {len(self.data_paths)}")
        self.transform = transform
        self.target_transform = target_transform
        self.hu_window = hu_window

    def _load_dicom(self, path, hu_window=None):
        """
        加载 DICOM 文件并处理。
        """
        dicom = pydicom.dcmread(path, force=True)
        img_array = dicom.pixel_array.astype(np.float32)

        if hu_window:
            hu_min, hu_max = hu_window
            img_array = np.clip(img_array, hu_min, hu_max)
            img_array = (img_array - hu_min) / (hu_max - hu_min)  # 归一化到 [0, 1]

        return img_array

    def __getitem__(self, index):
        img_path, label_path = self.data_paths[index]

        img_array = self._load_dicom(img_path, self.hu_window)
        label_array = self._load_dicom(label_path)

        img = Image.fromarray((img_array * 255).astype(np.uint8)).convert("L")
        label = Image.fromarray(label_array.astype(np.uint8)).convert("L")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.data_paths)
