import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, n_residual_blocks=3):
        super(Generator, self).__init__()

        # Initial convolution block
        self.model_head = nn.Sequential(
                      nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
                      nn.GroupNorm(8,64),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(64, 64, 3, stride=1, padding=1),
                      nn.GroupNorm(8,64),
                      nn.ReLU(inplace=True))

    
        self.model_down = nn.Sequential(
                                  nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                  nn.GroupNorm(8,128),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                  nn.GroupNorm(8,128),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                  nn.GroupNorm(8,256),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                  nn.GroupNorm(8,256),
                                  nn.ReLU(inplace=True),)

        # Residual blocks
        self.model_body = nn.Sequential(*[ResidualBlock(256) for _ in range(n_residual_blocks)],)

        self.model_up = nn.Sequential(
                                nn.Conv2d(256, 128, 3, stride=1, padding=1),
                                nn.GroupNorm(8,128),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                nn.GroupNorm(8,128),
                                nn.ReLU(inplace=True),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

                                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                                nn.GroupNorm(8,64),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                nn.GroupNorm(8,64),
                                nn.ReLU(inplace=True),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

                                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                                nn.GroupNorm(8,32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, 3, stride=1, padding=1),
                                nn.GroupNorm(8,32),
                                nn.ReLU(inplace=True),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                )
        
        self.model_tail = nn.Sequential(nn.Conv2d(32, 1, 3, stride=1, padding=1),
                                        nn.Tanh())
        
        # self.freeze_batchnorm()
        self._initialize_weights()

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
        # noise = torch.rand(1, 1, 64, 64).to(x.device)
        x = self.model_head(x)
        x = self.model_down(x)
        x = self.model_body(x)
        # x = torch.cat([x, noise], dim=1)
        x = self.model_up(x)
        x = self.model_tail(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, n_residual_blocks=3):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
                 nn.GroupNorm(8, 64),
                 nn.ReLU(inplace=True)]

        model += [nn.Conv2d(64, 128, 3, stride=2, padding=1),
                  nn.GroupNorm(8, 128),
                  nn.ReLU(inplace=True)]

        model += [nn.Conv2d(128, 256, 3, stride=2, padding=1),
                  nn.GroupNorm(8, 256),
                  nn.ReLU(inplace=True),
                  ]
        
        model += [nn.Conv2d(256, 512, 3, stride=2, padding=1),
                  nn.GroupNorm(8, 512),
                  nn.ReLU(inplace=True)]


        self.head = nn.Sequential(*model)
        # Residual blocks
        self.model_body = nn.Sequential(*[ResidualBlock(512) for _ in range(n_residual_blocks)],)
        self.model_tail = nn.Conv2d(512, 1, 1)
        self._initialize_weights()

    def forward(self, x):
        x = self.head(x)
        x = self.model_body(x)
        x = self.model_tail(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


    def _initialize_weights(self):
        # 遍历模型中的所有层，并应用He初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 对卷积层使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏置初始化为0


if __name__ == "__main__":
   
    model = Generator(in_channels=1, out_channels=1).to('cuda')
    input_tensor = torch.randn(1, 1, 512, 512).to('cuda')  
    import time
    s_time = time.time()
    output = model(input_tensor)
    e_time = time.time()
    print(output.shape)
    print(e_time - s_time)



