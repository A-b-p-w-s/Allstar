import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1, bias=False):
        super(DepthwiseConv2d, self).__init__()
        # Depthwise Convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                    padding=padding, groups=in_channels, bias=bias, dilation=dilation)
        
        # Pointwise Convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        return x

class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super(AttentionGate, self).__init__()
        self.conv_g = nn.Conv2d(gating_channels, in_channels, kernel_size=1)
        self.conv_x = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_psi = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Hardsigmoid(inplace=True)
    def forward(self, x, g):
        g = self.conv_g(g)
        x1 = self.conv_x(x)
        psi = self.relu(g + x1)
        psi = self.sigmoid(self.conv_psi(psi))
        return x * psi

class Dilation(nn.Module):
    def __init__(self, in_channels):
        super(Dilation, self).__init__()
        # Branch 1: standard convolution followed by batch normalization and activation
        self.branch1 = nn.Sequential(
            DepthwiseConv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # DepthwiseConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        )
        # Branch 2: dilated convolution with dilation factor 2
        self.branch2 = nn.Sequential(
            DepthwiseConv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        # Branch 3: dilated convolution with dilation factor 3
        self.branch3 = nn.Sequential(
            DepthwiseConv2d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        # Branch 4: dilated convolution with dilation factor 4
        self.branch4 = nn.Sequential(
            DepthwiseConv2d(in_channels, in_channels, kernel_size=3, dilation=4, padding=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        # Skip connection to match dimensions
        self.skip1 = nn.Sequential(nn.Conv2d(in_channels*4, in_channels, kernel_size=1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU())

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.skip1(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.down_sample(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up_sample = nn.Sequential(
            
        )

class AttResUNet(nn.Module):
    def __init__(self, in_channels, out_channels, mode='bilinear', flow=False):
        super(AttResUNet, self).__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU())
        self.enc1 = Dilation(64)
        self.enc2 = Dilation(128)
        self.enc3 = Dilation(128)
       
        if mode == 'bilinear':
            self.upconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                         nn.Conv2d(128, 128, kernel_size=1, stride=1))
            self.upconv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                         nn.Conv2d(128, 128, kernel_size=1, stride=1))
            self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                         nn.Conv2d(128, 64, kernel_size=1, stride=1))
        else:
            self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # self.gt1 = AttentionGate(256, 256)
        # self.gt2 = AttentionGate(128, 128)
        # self.gt3 = AttentionGate(64, 64)

        # Final output layer
        if flow:
            self.output_layer = nn.Sequential(
                nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
                nn.SiLU()
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )
        self.freeze_batchnorm()
        self._initialize_weights()

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
        x = self.input_layer(x)
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))

        x = self.upconv1(x)
        # dec = self.gt1(dec, enc2)
        x = self.upconv2(x)
        # dec = self.gt2(dec, enc1)
        x = self.upconv3(x)
        # dec = self.gt3(dec, inc)

        # Output layer
        return self.output_layer(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, mode='bilinear'):
        super(Discriminator, self).__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU())
        self.enc1 = Dilation(64, 128)
        self.enc2 = Dilation(128, 256)
        self.enc3 = Dilation(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.output_layer = nn.Sequential(
            nn.Conv2d(512, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.enc1(x)
        x = self.pool(x)

        x = self.enc2(x)
        x = self.pool(x)

        x = self.enc3(x)
        x = self.pool(x)

        x = self.output_layer(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        return x



if __name__ == "__main__":
   
    model = AttResUNet(in_channels=1, out_channels=1).to('cuda')
    input_tensor = torch.randn(1, 1, 512, 512).to('cuda')  
    import time
    s_time = time.time()
    output = model(input_tensor)
    e_time = time.time()
    print(output.shape)
    print(e_time - s_time)
