import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        
        # 用于匹配输入和输出的维度
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super(Encoder, self).__init__()
        self.enc1 = ResidualBlock(in_channels, base_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ResidualBlock(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x2 = self.enc2(x2)
        x3 = self.pool(x2)
        x3 = self.enc3(x3)
        x4 = self.pool(x3)
        x4 = self.enc4(x4)
        return x1, x2, x3, x4
        
class Decoder(nn.Module):
    def __init__(self, out_channels, base_channels=64):
        super(Decoder, self).__init__()
        self.upconv4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_channels * 8, base_channels * 4)
        
        self.upconv3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_channels * 4, base_channels * 2)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_channels * 2, out_channels)

        # self.upconv1 = nn.ConvTranspose2d(base_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2, x3, x4):
        x = self.upconv4(x4)
        x = torch.cat([x, x3], dim=1)  # skip connection
        x = self.dec4(x)

        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)  # skip connection
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)  # skip connection
        x = self.dec2(x)

        # x = self.upconv1(x)
        return x

class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32, n_residual_blocks=3):
        super(ResUNet, self).__init__()
        self.encoder = Encoder(in_channels, base_channels)
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock(base_channels*8, base_channels*8)]
        self.boddy = nn.Sequential(*model_body)
        self.decoder = Decoder(out_channels, base_channels)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x4 = self.boddy(x4)
        x = self.decoder(x1, x2, x3, x4)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 3, stride=1, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 2, stride=2),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 2, stride=2),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 2, stride=2),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 2, stride=2)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 1, 512, 512)
    # Example usage:
    in_channels = 1  # Number of input channels (e.g., RGB images)
    out_channels = 1  # Number of output classes (e.g., binary segmentation)
    model = ResUNet(in_channels, out_channels)
    dis = Discriminator(in_channels)
    # print(model)
    output = model(input_tensor)
    d_logit = dis(output)
    print(output.shape)
    print(d_logit.shape)