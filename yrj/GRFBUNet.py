from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class GRFB(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale=0.1, visual=12):
        super(GRFB, self).__init__()
        self.scale = scale
        inter_planes = in_channels // 8

        # Each branch contains only two BasicConv layers
        self.branch0 = nn.Sequential(
            BasicConv(in_channels, inter_planes, kernel_size=1, stride=stride),
            BasicConv(inter_planes, out_channels, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_channels, inter_planes, kernel_size=1, stride=stride),
            BasicConv(inter_planes, out_channels, kernel_size=3, stride=1, padding=2 * visual, dilation=2 * visual, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_channels, inter_planes, kernel_size=1, stride=stride),
            BasicConv(inter_planes, out_channels, kernel_size=3, stride=1, padding=3 * visual, dilation=3 * visual, relu=False)
        )

        self.ConvLinear = BasicConv(3 * out_channels, out_channels, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_channels, out_channels, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


# The rest of the model remains unchanged, using GRFB with optimized branches
class GRFBUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(GRFBUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = BasicConv(in_channels, base_c, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            GRFB(base_c, base_c * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            GRFB(base_c * 2, base_c * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            GRFB(base_c * 4, base_c * 8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            GRFB(base_c * 8, base_c * 16)
        )

        self.up1 = self._upsample_block(base_c * 16, base_c * 8, bilinear)
        self.up2 = self._upsample_block(base_c * 8, base_c * 4, bilinear)
        self.up3 = self._upsample_block(base_c * 4, base_c * 2, bilinear)
        self.up4 = self._upsample_block(base_c * 2, base_c, bilinear)
        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def _upsample_block(self, in_channels, out_channels, bilinear):
        if bilinear:
            up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        return nn.Sequential(
            up,
            GRFB(in_channels, out_channels)
        )

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.out_conv(x)
        return {"out": logits}
