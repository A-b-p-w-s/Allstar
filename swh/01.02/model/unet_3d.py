import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.se_3d import ChannelSpatialSELayer3D


class DoubleConv3D(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, num_groups=8):
        super(DoubleConv3D, self).__init__()
        self.layer = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channel, out_channel, kernel_size=3),
            nn.GroupNorm(num_groups, out_channel),
            nn.LeakyReLU(),
            nn.ReplicationPad3d(1),
            nn.Conv3d(out_channel, out_channel, kernel_size=3),
            nn.GroupNorm(num_groups, out_channel),
            nn.LeakyReLU())
    
    def forward(self, x):
        x = self.layer(x)
        return x

class encoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, num_groups=8):
        super(encoder, self).__init__()
        self.conv = DoubleConv3D(in_channel=in_channel, out_channel=out_channel, num_groups=num_groups)
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class decoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, num_groups=8):
        super(decoder, self).__init__()
        self.conv = DoubleConv3D(in_channel=in_channel, out_channel=out_channel, num_groups=num_groups)
        self.upsample = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReplicationPad3d(1),
                      nn.Conv3d(in_features, in_features, 3),
                      nn.GroupNorm(8, in_features),
                      nn.LeakyReLU(inplace=True),
                      nn.ReplicationPad3d(1),
                      nn.Conv3d(in_features, in_features, 3),
                      nn.GroupNorm(8, in_features),
                      nn.LeakyReLU(inplace=True),]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, num_groups=8):
        super(UNet3D, self).__init__()
        self.input_layer = nn.Sequential(
                            nn.ReplicationPad3d((3,3,3,3,1,1)),
                            nn.Conv3d(in_channel, 32, kernel_size=(3,7,7), stride=(1,1,1)),
                            nn.GroupNorm(4, 32),
                            nn.LeakyReLU())
        
        self.encoder1 = encoder(32, 32, num_groups=num_groups) 
        self.encoder2 = encoder(32, 64, num_groups=num_groups)
        self.encoder3 = encoder(64, 128, num_groups=num_groups)
        self.encoder4 = encoder(128, 256, num_groups=num_groups)
       
        self.decoder4 = decoder(256, 128, num_groups=num_groups)  # b, 8, 15, 1
        self.decoder3 = decoder(128*2, 64, num_groups=num_groups)  # b, 1, 28, 28
        self.decoder2 = decoder(64*2, 32, num_groups=num_groups)
        self.decoder1 = decoder(32*2, 32, num_groups=num_groups)

        self.output_layer = nn.Sequential(
                            nn.ReplicationPad3d((3,3,3,3,1,1)),
                            nn.Conv3d(32, out_channel, kernel_size=(3,7,7), stride=(1,1,1)),
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

# class Discriminator3D(nn.Module):
#     def __init__(self, in_channel=1, num_groups=8):
#         super(Discriminator3D, self).__init__()
#         self.input_layer = nn.Sequential(
#                     nn.ReplicationPad3d((3,3,3,3,1,1)),
#                     nn.Conv3d(in_channel, 16, kernel_size=(3,7,7), stride=(1,1,1)),
#                     nn.GroupNorm(8,16),
#                     nn.LeakyReLU())
        
#         self.encoder1 = encoder(16, 32, num_groups=num_groups) 
#         self.encoder2 = encoder(32, 64, num_groups=num_groups)
#         self.encoder3 = encoder(64, 128, num_groups=num_groups)
#         self.encoder4 = encoder(128, 256, num_groups=num_groups)
#         self._initialize_weights()
    
#     def forward(self, x):
#         x = self.input_layer(x)
#         x = self.encoder1(x)
#         x = self.encoder2(x)
#         x = self.encoder3(x)
#         x = self.encoder4(x)
#         return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)
    
#     def _initialize_weights(self):
#         # 遍历模型中的所有层，并应用He初始化
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):  # 对卷积层使用He初始化
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)  # 偏置初始化为0


class Discriminator3D(nn.Module):
    def __init__(self, in_channel=1, num_groups=8):
        super(Discriminator3D, self).__init__()
        input_layer = [ nn.ReplicationPad3d((3,3,3,3,1,1)),
                        nn.Conv3d(in_channel, 16, kernel_size=(3,7,7), stride=(1,1,1)),
                        nn.GroupNorm(8,16),
                        nn.LeakyReLU()]
        
        encoder1 = [encoder(16, 32, num_groups=num_groups)] 
        encoder2 = [encoder(32, 64, num_groups=num_groups)]
        encoder3 = [encoder(64, 128, num_groups=num_groups)]
        encoder4 = [encoder(128, 256, num_groups=num_groups)]
        model = input_layer + encoder1 + encoder2 + encoder3 + encoder4
        self.discriminator = nn.Sequential(*model)
        self._initialize_weights()
    
    def forward(self, x):
        x = self.discriminator(x)
        return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)
    
    def _initialize_weights(self):
        # 遍历模型中的所有层，并应用He初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):  # 对卷积层使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏置初始化为0


if __name__ == '__main__':
    data = torch.randn((1,1,1,64,64), requires_grad=False)

    print(f"data shape :{data.shape}")
    # net = UNet3D(1,1)
    # output = net(data)

    discriminator = Discriminator3D(in_channel=1,num_groups=8)
    output = discriminator(data)
    print(f"output shape: {output.shape}")