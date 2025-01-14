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
                      nn.Conv2d(in_channels, 256, 7, stride=2, padding=3),
                      nn.BatchNorm2d(256),
                      nn.ReLU(inplace=True))

    
        self.model_down = nn.Sequential(nn.Conv2d(256, 128, 3, stride=2, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 64, 3, stride=2, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))

        # Residual blocks
        self.model_body = nn.Sequential(*[ResidualBlock(64) for _ in range(n_residual_blocks)],)

        self.model_up = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(128, 256, kernel_size=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        self.model_tail = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1),
                                        nn.ReLU(inplace=True),
                                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(1, 1, kernel_size=3, padding=1),
                                        nn.Tanh())
        
        self.freeze_batchnorm()
        self._initialize_weights()


    def forward(self, x):
        noise = torch.rand(1, 1, 64, 64).to(x.device)
        x = self.model_head(x)
        x = self.model_down(x)
        x = self.model_body(x)
        x = torch.cat([x, noise], dim=1)
        x = self.model_up(x)
        x = self.model_tail(x)

        return x
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
    def __init__(self, in_channels, out_channels=1):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(in_channels, 256, 7, stride=2, padding=3),
                 nn.BatchNorm2d(256),
                 nn.ReLU(inplace=True)]

        model += [nn.Conv2d(256, 128, 3, stride=2, padding=1),
                  nn.BatchNorm2d(128),
                  nn.ReLU(inplace=True)]

        model += [nn.Conv2d(128, 64, 3, stride=2, padding=1),
                  nn.BatchNorm2d(64),
                  nn.ReLU(inplace=True)]

        # model += [nn.Conv2d(64, 32, 3, stride=2),
        #           nn.BatchNorm2d(32),
        #           nn.ReLU(inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(64, 1, 1)]

        self.model = nn.Sequential(*model)
        self.freeze_batchnorm()
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

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





