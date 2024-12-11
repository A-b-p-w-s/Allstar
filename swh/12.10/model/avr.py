import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, padding=0)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, padding=0)

    def forward(self, x):
        # avg_out = self.fc2(F.relu(self.fc1(self.global_avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.global_max_pool(x))))
        # out = avg_out + max_out
        return F.sigmoid(max_out) * x
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x_cat = torch.cat([avg_out, max_out], dim=1)
        # out = self.conv1(x_cat)
        return self.sigmoid(avg_out) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class up_sampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_sampling, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
    def forward(self, x):
        return self.up(x)

class AVR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AVR, self).__init__()
        self.up1 = up_sampling(in_channels, 1024)
        self.up2 = up_sampling(1024, 512)
        self.up3 = up_sampling(512, 256)
        self.up4 = up_sampling(256, 128)
        self.output = nn.Sequential(
                    nn.Conv2d(128, 64, kernel_size=1, stride=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.Tanh())
        # self.cbam0 = CBAM(2048)
        self.cbam1 = CBAM(1024)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(128)

        self._initialize_weights()
        # self.freeze_batchnorm()

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                # 将 running_mean 和 running_var 设置为固定值
                m.running_mean.zero_()  # 运行均值设为0
                m.running_var.fill_(1)   # 运行方差设为1

                # 冻结 BatchNorm 层的权重和偏置
                m.weight.data.fill_(1)  # 权重设为1
                m.bias.data.zero_()     # 偏置设为0

                # 将 BatchNorm 设置为评估模式，使用固定的统计信息
                m.eval()

                # 冻结参数，不更新梯度
                for param in m.parameters():
                    param.requires_grad = False

    def _initialize_weights(self):
        # 遍历模型中的所有层，并应用He初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 对卷积层使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对全连接层使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏置初始化为0

    def forward(self, x):
        # x = self.cbam0(x)
        x = self.up1(x)
        x = self.cbam1(x)
        x = self.up2(x)
        x = self.cbam2(x)
        x = self.up3(x)
        x = self.cbam3(x)
        x = self.up4(x)
        x = self.cbam4(x)
        x = self.output(x)
        return x
       
if __name__ == '__main__':
    x = torch.randn(1, 1, 16, 16)

    # 定义反卷积层，假设我们将特征图的大小从 8x8 放大到 16x16
    model = AVR(1, 1)

    # 输出大小将是 (1, 1, 512, 512)
    output = model(x)
    print(output.shape)  # torch.Size([1, 1, 512, 512])
