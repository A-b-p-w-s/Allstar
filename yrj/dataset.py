import torch.utils.data as data
import PIL.Image as Image
import os

class VesselDataset(data.Dataset):
    def __init__(self,root,transform=None,target_transform=None):
        self.data_paths=[]
        n=len(os.listdir(root))//2
        for i in range(n):
            img_path=os.path.join(root,f"{i:05d}.png")
            mask_path=os.path.join(root,f"{i:05d}_mask.png")
            self.data_paths.append((img_path,mask_path))
        self.transform=transform
        self.target_transform=target_transform

    def __getitem__(self, index):
        paths=self.data_paths[index]
        origin_x=Image.open(paths[0])
        origin_y=Image.open(paths[1]).convert("L")
        if self.transform is not None:
            img=self.transform(origin_x)
        if self.target_transform is not None:
            mask=self.target_transform(origin_y)
        return img,mask

    def __len__(self):
        return len(self.data_paths)

    