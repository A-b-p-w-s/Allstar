import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, att_heads=8):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, att_heads, 1, padding=0)
        self.fc2 = nn.Conv2d(att_heads, in_channels, 1, padding=0)

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
    def __init__(self, in_channels, att_heads=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, att_heads)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.GroupNorm(out_channels//num_groups, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.GroupNorm(out_channels//num_groups, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_groups=8):
        super(encoder, self).__init__()
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, num_groups=num_groups)
        self.cbam = CBAM(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.double_conv(x)
        x = self.cbam(x)
        x = self.pool(x)
        return x

class decoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_groups=8):
        super(decoder, self).__init__()
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, num_groups=num_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        x = self.double_conv(x)
        x = self.upsample(x)
        return x

class down_sample(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_groups=8):
        super(down_sample, self).__init__()
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, num_groups=num_groups)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.double_conv(x)
        x = self.pool(x)
        return x
    
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_groups=8):
        super(Generator, self).__init__()
        self.input_layer = nn.Sequential(
                            nn.ReflectionPad2d(3),
                            nn.Conv2d(in_channels, 64, kernel_size=7),
                            nn.GroupNorm(num_groups, 64),
                            nn.LeakyReLU())
        
        self.encoder1 = encoder(64, 64, num_groups=num_groups) 
        self.encoder2 = encoder(64, 128, num_groups=num_groups)
        self.encoder3 = encoder(128, 256, num_groups=num_groups)
        self.encoder4 = encoder(256, 512, num_groups=num_groups)
       
        self.decoder4 = decoder(512, 256, num_groups=num_groups)  
        self.decoder3 = decoder(256*2, 128, num_groups=num_groups)  
        self.decoder2 = decoder(128*2, 64, num_groups=num_groups)
        self.decoder1 = decoder(64*2, 64, num_groups=num_groups)

        self.output_layer = nn.Sequential(
                            nn.ReflectionPad2d(3),
                            nn.Conv2d(64, out_channels, kernel_size=7, stride=1),
                            nn.Sigmoid())
        
        self._initialize_weights()
        

    def forward(self, x):
        x = self.input_layer(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        out = self.decoder4(x4)
        out = self.decoder3(torch.cat((x3,out), dim=1))
        out = self.decoder2(torch.cat((x2,out), dim=1))
        out = self.decoder1(torch.cat((x1,out), dim=1))
        out = self.output_layer(out)
        
        return out
    
    def _initialize_weights(self):
        # 遍历模型中的所有层，并应用He初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):  # 对卷积层使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏置初始化为0
        
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_groups=8):
        super(Discriminator, self).__init__()
        self.input_layer = nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels, 64, kernel_size=7),
                    nn.GroupNorm(8,64),
                    nn.LeakyReLU())
        
        self.encoder1 = down_sample(64, 64, num_groups=num_groups) 
        self.encoder2 = down_sample(64, 128, num_groups=num_groups)
        self.encoder3 = down_sample(128, 256, num_groups=num_groups)
        self.encoder4 = down_sample(256, 512, num_groups=num_groups)
        self._initialize_weights()
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    
    def _initialize_weights(self):
        # 遍历模型中的所有层，并应用He初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):  # 对卷积层使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏置初始化为0


def test_unet_with_cbam():
    # 模拟一个随机的输入，假设输入图像大小为 (batch_size, channels, height, width)
    batch_size = 2
    in_channels = 1  
    out_channels = 1  
    height, width = 256, 256  

    x = torch.randn(batch_size, in_channels, height, width)

    model = Generator(in_channels=in_channels, out_channels=out_channels)

    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (batch_size, out_channels, height, width), \
        f"Expected output shape {(batch_size, out_channels, height, width)}, but got {output.shape}"

    print("Test passed!")

if __name__ == "__main__":
    test_unet_with_cbam()