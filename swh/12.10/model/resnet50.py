import torch
import torch.nn as nn
import torchvision.models as models


# resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 去除最后的展平层（avgpool）和全连接层（fc）
class ResNet50Backbone(nn.Module):
    def __init__(self):
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        super(ResNet50Backbone, self).__init__()
        # 取出除了最后的avgpool和fc层之外的部分
        self.features = nn.Sequential(*list(resnet50.children())[:-2])
        
        # 冻结所有层的参数
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        return x


class ResNet50layers(nn.Module):
    def __init__(self):
        super(ResNet50layers, self).__init__()
        # 加载预训练的ResNet50
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 获取不同的中间层输出
        self.layer1 = self.resnet50.layer1
        self.layer2 = self.resnet50.layer2
        self.layer3 = self.resnet50.layer3
        self.layer4 = self.resnet50.layer4
        
        # 修改ResNet50的输出结构，仅保留特征提取部分
        self.input_layer = nn.Sequential(*list(self.resnet50.children())[:-2])[0]
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        # 通过ResNet50提取特征
        x = self.input_layer(x)
        x1 = self.layer1(x)   # output from layer1
        x2 = self.layer2(x1)  # output from layer2
        x3 = self.layer3(x2)  # output from layer3
        x4 = self.layer4(x3)  # output from layer4
        
        return x1, x2, x3, x4



if __name__ == '__main__':
    # 创建新的网络
    backbone = ResNet50Backbone()

    print(backbone)

    data = torch.randn(1, 3, 512, 512)
    # out = backbone(data) # [1,2048,16,16]
    # print(out.shape)
    layers = ResNet50layers()
    out = layers(data)
