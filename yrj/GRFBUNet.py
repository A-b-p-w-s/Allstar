# from typing import Dict
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class BasicConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU(inplace=True) if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x


# class GRFB(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, scale=0.1, visual=12):
#         super(GRFB, self).__init__()
#         self.scale = scale
#         inter_planes = in_channels // 8

#         # Each branch contains only two BasicConv layers
#         self.branch0 = nn.Sequential(
#             BasicConv(in_channels, inter_planes, kernel_size=1, stride=stride),
#             BasicConv(inter_planes, out_channels, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv(in_channels, inter_planes, kernel_size=1, stride=stride),
#             BasicConv(inter_planes, out_channels, kernel_size=3, stride=1, padding=2 * visual, dilation=2 * visual, relu=False)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv(in_channels, inter_planes, kernel_size=1, stride=stride),
#             BasicConv(inter_planes, out_channels, kernel_size=3, stride=1, padding=3 * visual, dilation=3 * visual, relu=False)
#         )

#         self.ConvLinear = BasicConv(3 * out_channels, out_channels, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv(in_channels, out_channels, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)

#         out = torch.cat((x0, x1, x2), 1)
#         out = self.ConvLinear(out)
#         short = self.shortcut(x)
#         out = out * self.scale + short
#         out = self.relu(out)

#         return out


# # The rest of the model remains unchanged, using GRFB with optimized branches
# class GRFBUNet(nn.Module):
#     def __init__(self,
#                  in_channels: int = 1,
#                  num_classes: int = 2,
#                  bilinear: bool = True,
#                  base_c: int = 64):
#         super(GRFBUNet, self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.bilinear = bilinear

#         self.in_conv = BasicConv(in_channels, base_c, kernel_size=3, padding=1)
#         self.down1 = nn.Sequential(
#             nn.MaxPool2d(2),
#             GRFB(base_c, base_c * 2)
#         )
#         self.down2 = nn.Sequential(
#             nn.MaxPool2d(2),
#             GRFB(base_c * 2, base_c * 4)
#         )
#         self.down3 = nn.Sequential(
#             nn.MaxPool2d(2),
#             GRFB(base_c * 4, base_c * 8)
#         )
#         self.down4 = nn.Sequential(
#             nn.MaxPool2d(2),
#             GRFB(base_c * 8, base_c * 16)
#         )

#         self.up1 = self._upsample_block(base_c * 16, base_c * 8, bilinear)
#         self.up2 = self._upsample_block(base_c * 8, base_c * 4, bilinear)
#         self.up3 = self._upsample_block(base_c * 4, base_c * 2, bilinear)
#         self.up4 = self._upsample_block(base_c * 2, base_c, bilinear)
#         self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

#     def _upsample_block(self, in_channels, out_channels, bilinear):
#         if bilinear:
#             up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#         return nn.Sequential(
#             up,
#             GRFB(in_channels, out_channels)
#         )

#     def forward(self, x):
#         x1 = self.in_conv(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5)
#         x = self.up2(x)
#         x = self.up3(x)
#         x = self.up4(x)
#         logits = self.out_conv(x)
#         output = nn.Sigmoid()(logits)
#         return output
    
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import  List


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DoubleConv1(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv1, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            GRFB(mid_channels, out_channels, stride=1, scale=0.1, visual=12)
           

        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv1(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class BasicConv(nn.Module):

        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                     bn=True, bias=False):
            super(BasicConv, self).__init__()
            self.out_channels = out_channels
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
            self.out_channels = out_channels
            inter_planes = in_channels // 8
            self.branch0 = nn.Sequential(
                BasicConv(in_channels, 2 * inter_planes, kernel_size=1, stride=stride),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
                          relu=False),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride)
            )
            self.branch1 = nn.Sequential(
                BasicConv(in_channels, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=inter_planes),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual,
                          dilation=2 * visual, relu=False),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=1)
            )
            self.branch2 = nn.Sequential(
                BasicConv(in_channels, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, groups=inter_planes),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=2 * inter_planes),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3 * visual,
                          dilation=3 * visual, relu=False),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride)
            )

            self.ConvLinear = BasicConv(14 * inter_planes, out_channels, kernel_size=1, stride=1, relu=False)
            self.shortcut = BasicConv(in_channels, out_channels, kernel_size=1, stride=stride, relu=False)
            self.relu = nn.ReLU(inplace=False)

        def forward(self, x):
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)

            out = torch.cat((x, x0, x1, x2), 1)
            out = self.ConvLinear(out)
            short = self.shortcut(x)
            out = out * self.scale + short
            out = self.relu(out)

            return out


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

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        output = nn.Sigmoid()(logits)
        return output

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # CBAM Implementation
# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
#         self.spatial_attention = SpatialAttention(kernel_size)

#     def forward(self, x):
#         x = self.channel_attention(x) * x
#         x = self.spatial_attention(x) * x
#         return x


# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv(x)
#         return self.sigmoid(x)


# # Attention Gate Implementation
# class AttentionGate(nn.Module):
#     def __init__(self, in_channels, gating_channels, inter_channels):
#         super(AttentionGate, self).__init__()
#         self.Wx = nn.Sequential(
#             nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(inter_channels)
#         )
#         self.Wg = nn.Sequential(
#             nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(inter_channels)
#         )
#         self.psi = nn.Sequential(
#             nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x, g):
#         x1 = self.Wx(x)
#         g1 = self.Wg(g)
#         psi = self.relu(x1 + g1)
#         psi = self.psi(psi)
#         return x * psi


# # Basic Convolution Block
# class BasicConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, relu=True):
#         super(BasicConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True) if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x


# # GRFB Module with CBAM
# class GRFB(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, scale=0.1, visual=12):
#         super(GRFB, self).__init__()
#         self.scale = scale
#         inter_planes = in_channels // 8
#         self.branch0 = nn.Sequential(
#             BasicConv(in_channels, inter_planes, kernel_size=1, stride=stride),
#             BasicConv(inter_planes, out_channels, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv(in_channels, inter_planes, kernel_size=1, stride=stride),
#             BasicConv(inter_planes, out_channels, kernel_size=3, stride=1, padding=2 * visual, dilation=2 * visual, relu=False)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv(in_channels, inter_planes, kernel_size=1, stride=stride),
#             BasicConv(inter_planes, out_channels, kernel_size=3, stride=1, padding=3 * visual, dilation=3 * visual, relu=False)
#         )

#         self.ConvLinear = BasicConv(3 * out_channels, out_channels, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv(in_channels, out_channels, kernel_size=1, stride=stride, relu=False)
#         self.cbam = CBAM(out_channels)  # Add CBAM
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)

#         out = torch.cat((x0, x1, x2), 1)
#         out = self.ConvLinear(out)
#         short = self.shortcut(x)
#         out = out * self.scale + short
#         out = self.cbam(out)  # Apply CBAM
#         out = self.relu(out)

#         return out


# # Up Module with Attention Gate
# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super(Up, self).__init__()
#         self.attention_gate = AttentionGate(out_channels, in_channels // 2, in_channels // 4)
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         x1 = self.up(x1)
#         # Apply Attention Gate
#         x2 = self.attention_gate(x2, x1)
#         # Padding to match dimensions
#         diff_y = x2.size()[2] - x1.size()[2]
#         diff_x = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
#                         diff_y // 2, diff_y - diff_y // 2])
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x


# # Double Convolution Block
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super(DoubleConv, self).__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# # Main MultiResUNet Model with Attention
# class MultiResUNetAttention(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(MultiResUNetAttention, self).__init__()
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = GRFB(64, 128)
#         self.down2 = GRFB(128, 256)
#         self.down3 = GRFB(256, 512)
#         self.down4 = GRFB(512, 1024)
#         self.up1 = Up(1024, 512)
#         self.up2 = Up(512, 256)
#         self.up3 = Up(256, 128)
#         self.up4 = Up(128, 64)
#         self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
