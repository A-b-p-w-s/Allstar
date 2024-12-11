import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer_2D(nn.Module):
    def __init__(self):
        super(Transformer_2D, self).__init__()

    # @staticmethod
    def forward(self, src, flow):
        """
        前向传播方法，根据流场对源图像进行变形。

        参数:
        - src: 源图像张量，形状为 (b, c, h, w)
        - flow: 流场张量，形状为 (b, 2, h, w)

        返回:
        - warped: 变形后的图像张量，形状为 (b, c, h, w)
        """
        b, _, h, w = flow.shape
        device, dtype = flow.device, flow.dtype

        # 构造标准网格
        vectors = [torch.arange(0, s, device=device, dtype=dtype) for s in (h, w)]
        grids = torch.meshgrid(vectors, indexing="ij")  # indexing='ij' -> (y, x) 顺序
        grid = torch.stack(grids)  # (2, h, w)
        grid = grid.repeat(b, 1, 1, 1)  # (b, 2, h, w)

        # 计算新的坐标
        new_locs = grid + flow
        new_locs[:, 0, ...] = 2 * (new_locs[:, 0, ...] / (h - 1) - 0.5)  # 归一化 y
        new_locs[:, 1, ...] = 2 * (new_locs[:, 1, ...] / (w - 1) - 0.5)  # 归一化 x

        # 调整维度顺序以匹配 grid_sample 要求
        new_locs = new_locs.permute(0, 2, 3, 1)  # (b, h, w, 2)
        new_locs = new_locs[..., [1 , 0]]
        
        # 使用 grid_sample 进行插值
        warped = F.grid_sample(src, new_locs, align_corners=True, padding_mode="border")

        return warped
