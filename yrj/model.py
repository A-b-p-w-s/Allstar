# import torch
# import torch.nn as nn

# class DoubleConv(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         super(DoubleConv,self).__init__()
#         self.conv=nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
#                                 nn.BatchNorm2d(out_ch),
#                                 nn.ReLU(inplace=True),
#                                 nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1),
#                                 nn.BatchNorm2d(out_ch),
#                                 nn.ReLU(inplace=True))
#         self.se = SEBlock(out_ch)

#     def forward(self,input):
#         input = self.conv(input)
#         input = self.se(input)
#         return input
    
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.fc1 = nn.Linear(channels, channels // reduction)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(channels // reduction, channels)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         batch, channels, _, _ = x.size()
#         y = x.view(batch, channels, -1).mean(dim=2)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y).view(batch, channels, 1, 1)
#         return x * y

# class LightDoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),  # 深度卷积
#             nn.BatchNorm2d(in_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_ch, out_ch, 1),  # 逐点卷积
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),  # 深度卷积
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 1),  # 逐点卷积
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, input):
#         return self.conv(input)

# class Unet(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         super(Unet,self).__init__()

#         self.conv1=DoubleConv(in_ch,64)
#         self.pool1=nn.MaxPool2d(2) #256
#         self.conv2=DoubleConv(64,128)
#         self.pool2=nn.MaxPool2d(2) #128
#         self.conv3=DoubleConv(128,256)
#         self.pool3=nn.MaxPool2d(2) #64
#         self.conv4=DoubleConv(256,512)
#         self.pool4=nn.MaxPool2d(2) #32
#         self.conv5=DoubleConv(512,1024)
#         self.up6=nn.ConvTranspose2d(1024,512,2,stride=2) #64
#         self.conv6=DoubleConv(1024,512)
#         self.up7=nn.ConvTranspose2d(512,256,2,stride=2)#128
#         self.conv7=DoubleConv(512,256)
#         self.up8=nn.ConvTranspose2d(256,128,2,stride=2)#256
#         self.conv8=DoubleConv(256,128)
#         self.up9=nn.ConvTranspose2d(128,64,2,stride=2)#512
#         self.conv9=DoubleConv(128,64)
#         self.conv10=nn.Conv2d(64,out_ch,1)


#     def forward(self,x):
#         c1=self.conv1(x)
#         p1=self.pool1(c1)
#         c2 = self.conv2(p1)
#         p2 = self.pool2(c2)
#         c3 = self.conv3(p2)
#         p3 = self.pool3(c3)
#         c4 = self.conv4(p3)
#         p4 = self.pool4(c4)
#         c5=self.conv5(p4)

#         u6=self.up6(c5)
#         merge6=torch.cat([u6,c4],dim=1)
#         c6=self.conv6(merge6)
#         u7=self.up7(c6)
#         merge7=torch.cat([u7,c3],dim=1)
#         c7=self.conv7(merge7)
#         u8=self.up8(c7)
#         merge8=torch.cat([u8,c2],dim=1)
#         c8=self.conv8(merge8)
#         u9=self.up9(c8)
#         merge9=torch.cat([u9,c1],dim=1)
#         c9=self.conv9(merge9)
#         c10=self.conv10(c9)
#         out=nn.Sigmoid()(c10)
#         return out

# class LightUnet(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         super(LightUnet,self).__init__()

#         self.conv1=LightDoubleConv(in_ch,64)
#         self.pool1=nn.MaxPool2d(2) #256
#         self.conv2=LightDoubleConv(64,128)
#         self.pool2=nn.MaxPool2d(2) #128
#         self.conv3=LightDoubleConv(128,256)
#         self.pool3=nn.MaxPool2d(2) #64
#         self.conv4=LightDoubleConv(256,512)
#         self.pool4=nn.MaxPool2d(2) #32
#         self.conv5=LightDoubleConv(512,1024)
#         self.up6=nn.ConvTranspose2d(1024,512,2,stride=2) #64
#         self.conv6=LightDoubleConv(1024,512)
#         self.up7=nn.ConvTranspose2d(512,256,2,stride=2)#128
#         self.conv7=LightDoubleConv(512,256)
#         self.up8=nn.ConvTranspose2d(256,128,2,stride=2)#256
#         self.conv8=LightDoubleConv(256,128)
#         self.up9=nn.ConvTranspose2d(128,64,2,stride=2)#512
#         self.conv9=LightDoubleConv(128,64)
#         self.conv10=nn.Conv2d(64,out_ch,1)


#     def forward(self,x):
#         c1=self.conv1(x)
#         p1=self.pool1(c1)
#         c2 = self.conv2(p1)
#         p2 = self.pool2(c2)
#         c3 = self.conv3(p2)
#         p3 = self.pool3(c3)
#         c4 = self.conv4(p3)
#         p4 = self.pool4(c4)
#         c5=self.conv5(p4)

#         u6=self.up6(c5)
#         merge6=torch.cat([u6,c4],dim=1)
#         c6=self.conv6(merge6)
#         u7=self.up7(c6)
#         merge7=torch.cat([u7,c3],dim=1)
#         c7=self.conv7(merge7)
#         u8=self.up8(c7)
#         merge8=torch.cat([u8,c2],dim=1)
#         c8=self.conv8(merge8)
#         u9=self.up9(c8)
#         merge9=torch.cat([u9,c1],dim=1)
#         c9=self.conv9(merge9)
#         c10=self.conv10(c9)
#         out=nn.Sigmoid()(c10)
#         return out
    
import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        sa = self.spatial_attention(torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1))
        x = x * sa
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UnetWithAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnetWithAttention, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att6 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att7 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att8 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att9 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        u6 = self.up6(c5)
        c4 = self.att6(u6, c4)
        merge6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(merge6)

        u7 = self.up7(c6)
        c3 = self.att7(u7, c3)
        merge7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(merge7)

        u8 = self.up8(c7)
        c2 = self.att8(u8, c2)
        merge8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(merge8)

        u9 = self.up9(c8)
        c1 = self.att9(u9, c1)
        merge9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out
