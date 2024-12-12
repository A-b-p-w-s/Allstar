import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, padding=0)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, padding=0)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.global_avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.global_max_pool(x))))
        out = avg_out + max_out
        return F.sigmoid(out) * x
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseConv2d, self).__init__()
        # Depthwise Convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, groups=in_channels, bias=bias, dilation=dilation)
        
        # Pointwise Convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        return x

class down_sampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_sampling, self).__init__()
        self.down = nn.Sequential(
            DepthwiseConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseConv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.down(x)
        
class up_sampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_sampling, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
    def forward(self, x):
        return self.up(x)

class UNet_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, flow=False):
        super(UNet_CBAM, self).__init__()
        self.flow = flow
        # 编码器部分
        self.input_layer = nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=7, stride=2, padding=3),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU())
        self.enc1 = down_sampling(256, 128)
        self.enc2 = down_sampling(128, 64)
        # self.enc3 = down_sampling(64, 32)
        # self.enc4 = down_sampling(256, 512)
        
        # 解码器部分
        self.up3 = up_sampling(64, 128)
        self.up2 = up_sampling(128*2, 256)
        # self.up1 = up_sampling(128*2, 64)
        # self.up1 = up_sampling(64*2, 64)
        
        # Final output layer
        if flow:
            self.output_layer = nn.Sequential(
                nn.Conv2d(64, out_channels, kernel_size=1),
                nn.SiLU()
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Conv2d(256*2, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Tanh())
        
        # CBAM模块
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(64)
        # self.cbam4 = CBAM(512)
        self._initialize_weights()
        self.freeze_batchnorm()
        
    def forward(self, x):
        # 编码器阶段
        x = self.input_layer(x) # 256
        enc1 = self.enc1(x) # 128
        enc2 = self.enc2(enc1) # 64
        # enc3 = self.enc3(enc2) # 512
        # enc4 = self.enc4(enc3)
        
        # CBAM增强特征
        x = self.cbam1(x) # 256
        enc1 = self.cbam2(enc1) # 128
        enc2 = self.cbam3(enc2) # 64
        # enc4 = self.cbam4(enc4)
        
        # 解码器阶段
        dec = self.up3(enc2) # 128
        dec = self.up2(torch.cat((enc1,dec), dim=1)) #256
        # dec = self.up1(torch.cat((enc1,dec), dim=1))
        # dec = self.up1(dec)
        dec = self.output_layer(torch.cat((x,dec), dim=1))
        return dec
    
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

    
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Discriminator, self).__init__()
        self.input_layer = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3)
        self.down1 = down_sampling(64, 128)
        self.down2 = down_sampling(128, 256)
        self.down3 = down_sampling(256, 512)
        # self.down4 = down_sampling(512, 512)
        self.output_layer = nn.Sequential(
            nn.Conv2d(512, out_channels, kernel_size=1, stride=1),
            )

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

    def forward(self, x):
        x = self.input_layer(x)

        x = self.cbam1(x)
        x = self.down1(x)

        x = self.cbam2(x)
        x = self.down2(x)

        x = self.cbam3(x)
        x = self.down3(x)

        x = self.cbam4(x)
        # x = self.down4(x)

        x = self.output_layer(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        return x

class Discriminator2(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(Discriminator2, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(in_channels, 64, 7, stride=1, padding=3),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True)]

        model += [nn.Conv2d(64, 128, 3, stride=2, padding=1),
                  nn.BatchNorm2d(128),
                  nn.ReLU(inplace=True)]

        model += [nn.Conv2d(128, 256, 3, stride=2, padding=1),
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True)]

        model += [nn.Conv2d(256, 512, 3, stride=2, padding=1),
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, kernel_size=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

def test_unet_with_cbam():
    # 模拟一个随机的输入，假设输入图像大小为 (batch_size, channels, height, width)
    batch_size = 2
    in_channels = 3  
    out_channels = 1  
    height, width = 256, 256  

    x = torch.randn(batch_size, in_channels, height, width)

    model = UNet_CBAM(in_channels=in_channels, out_channels=out_channels)

    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (batch_size, out_channels, height, width), \
        f"Expected output shape {(batch_size, out_channels, height, width)}, but got {output.shape}"

    print("Test passed!")

if __name__ == "__main__":
    test_unet_with_cbam()