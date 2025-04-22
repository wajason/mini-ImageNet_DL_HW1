import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights  # 導入 ResNet18_Weights

class NaiveModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=50, num_layers=5):
        super(NaiveModel, self).__init__()
        # 載入預訓練的 ResNet-18
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 修改第一層以適應不同的輸入通道數（in_channels）
        self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, bias=False)
        resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 提取 ResNet 的前幾層（直到 layer4），作為特徵提取器
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64 通道
            resnet.layer2,  # 128 通道
            resnet.layer3,  # 256 通道
            resnet.layer4   # 512 通道
        )
        
        # 使用標準卷積層（nn.Conv2d），與 TaskAModel 的層數和通道數一致
        self.conv_layers = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        channels = [512, 256, 128][:num_layers]  # 與 TaskAModel 一致
        
        for i in range(len(channels)-1):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU()
                )
            )
            if channels[i] != channels[i+1]:
                self.residual_convs.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=1, stride=1))
            else:
                self.residual_convs.append(None)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(channels[-1], num_classes)
        self.num_layers = len(channels) - 1

    def forward(self, x):
        # 通道適配
        x = self.channel_adapter(x)
        # 通過 ResNet 特徵提取器
        x = self.feature_extractor(x)
        
        # 通過標準卷積層
        for i, layer in enumerate(self.conv_layers):
            identity = x
            x = layer(x)
            if self.residual_convs[i] is not None:
                identity = self.residual_convs[i](identity)
            x = x + identity
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x