import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DoubleConv,self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1),
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU(inplace=True))
        self.se = SEBlock(out_ch)

    def forward(self,input):
        input = self.conv(input)
        input = self.se(input)
        return input
    
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = x.view(batch, channels, -1).mean(dim=2)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y

class LightDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),  # 深度卷积
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),  # 逐点卷积
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),  # 深度卷积
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),  # 逐点卷积
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet,self).__init__()

        self.conv1=DoubleConv(in_ch,64)
        self.pool1=nn.MaxPool2d(2) #256
        self.conv2=DoubleConv(64,128)
        self.pool2=nn.MaxPool2d(2) #128
        self.conv3=DoubleConv(128,256)
        self.pool3=nn.MaxPool2d(2) #64
        self.conv4=DoubleConv(256,512)
        self.pool4=nn.MaxPool2d(2) #32
        self.conv5=DoubleConv(512,1024)
        self.up6=nn.ConvTranspose2d(1024,512,2,stride=2) #64
        self.conv6=DoubleConv(1024,512)
        self.up7=nn.ConvTranspose2d(512,256,2,stride=2)#128
        self.conv7=DoubleConv(512,256)
        self.up8=nn.ConvTranspose2d(256,128,2,stride=2)#256
        self.conv8=DoubleConv(256,128)
        self.up9=nn.ConvTranspose2d(128,64,2,stride=2)#512
        self.conv9=DoubleConv(128,64)
        self.conv10=nn.Conv2d(64,out_ch,1)


    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5=self.conv5(p4)

        u6=self.up6(c5)
        merge6=torch.cat([u6,c4],dim=1)
        c6=self.conv6(merge6)
        u7=self.up7(c6)
        merge7=torch.cat([u7,c3],dim=1)
        c7=self.conv7(merge7)
        u8=self.up8(c7)
        merge8=torch.cat([u8,c2],dim=1)
        c8=self.conv8(merge8)
        u9=self.up9(c8)
        merge9=torch.cat([u9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out=nn.Sigmoid()(c10)
        return out

class LightUnet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(LightUnet,self).__init__()

        self.conv1=LightDoubleConv(in_ch,64)
        self.pool1=nn.MaxPool2d(2) #256
        self.conv2=LightDoubleConv(64,128)
        self.pool2=nn.MaxPool2d(2) #128
        self.conv3=LightDoubleConv(128,256)
        self.pool3=nn.MaxPool2d(2) #64
        self.conv4=LightDoubleConv(256,512)
        self.pool4=nn.MaxPool2d(2) #32
        self.conv5=LightDoubleConv(512,1024)
        self.up6=nn.ConvTranspose2d(1024,512,2,stride=2) #64
        self.conv6=LightDoubleConv(1024,512)
        self.up7=nn.ConvTranspose2d(512,256,2,stride=2)#128
        self.conv7=LightDoubleConv(512,256)
        self.up8=nn.ConvTranspose2d(256,128,2,stride=2)#256
        self.conv8=LightDoubleConv(256,128)
        self.up9=nn.ConvTranspose2d(128,64,2,stride=2)#512
        self.conv9=LightDoubleConv(128,64)
        self.conv10=nn.Conv2d(64,out_ch,1)


    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5=self.conv5(p4)

        u6=self.up6(c5)
        merge6=torch.cat([u6,c4],dim=1)
        c6=self.conv6(merge6)
        u7=self.up7(c6)
        merge7=torch.cat([u7,c3],dim=1)
        c7=self.conv7(merge7)
        u8=self.up8(c7)
        merge8=torch.cat([u8,c2],dim=1)
        c8=self.conv8(merge8)
        u9=self.up9(c8)
        merge9=torch.cat([u9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out=nn.Sigmoid()(c10)
        return out