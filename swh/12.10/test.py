import torch
import torch.nn as nn

# 输入：一个大小为 (batch_size=1, in_channels=3, height=8, width=8) 的张量
x = torch.randn(1, 1, 8, 8)

# 定义反卷积层，假设我们将特征图的大小从 8x8 放大到 16x16
conv_transpose = nn.ConvTranspose2d(1, 3, kernel_size=2, stride=2, padding=0)

# 输出大小将是 (1, 6, 16, 16)
output = conv_transpose(x)
print(output.shape)  # torch.Size([1, 6, 16, 16])
