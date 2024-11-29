import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_classes=3, img_height=512, img_width=512, img_channels=3):
        super(UNet, self).__init__()

        # Contracting path
        self.c1 = self.contract_block(img_channels, 16)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = self.contract_block(16, 32)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = self.contract_block(32, 64)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c4 = self.contract_block(64, 128)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c5 = self.contract_block(128, 256)

        # Expanding path
        self.u6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c6 = self.expand_block(256, 128)
        self.u7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c7 = self.expand_block(128, 64)
        self.u8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c8 = self.expand_block(64, 32)
        self.u9 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.c9 = self.expand_block(32, 16)

        self.outputs = nn.Conv2d(16, n_classes, kernel_size=1)

    def contract_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def expand_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def forward(self, x):
        # Contracting path
        c1 = self.c1(x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        c3 = self.c3(p2)
        p3 = self.p3(c3)
        c4 = self.c4(p3)
        p4 = self.p4(c4)
        c5 = self.c5(p4)

        # Expanding path
        u6 = self.u6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6(u6)
        u7 = self.u7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7(u7)
        u8 = self.u8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8(u8)
        u9 = self.u9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9(u9)

        outputs = self.outputs(c9)
        outputs = F.softmax(outputs, dim=1)

        return outputs

# Creating an instance of the UNet model
unet_model = UNet()

# Printing the model summary
print(unet_model)
'''
# Plotting the model (if you have graphviz installed)
from torchviz import make_dot
dummy_input = torch.randn(1, 3, 512, 512)  # Assuming input shape (batch_size, channels, height, width)
make_dot(unet_model(dummy_input), params=dict(unet_model.named_parameters())).render("UNet_Model", format="png")
'''