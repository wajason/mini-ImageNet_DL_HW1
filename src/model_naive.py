import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class NaiveModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=50, num_layers=5):
        super(NaiveModel, self).__init__()
        self.layers = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        channels = [in_channels, 32, 64, 128, 256, 512][:num_layers+1]

        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU()
                )
            )
            if i > 0 and channels[i] != channels[i+1]:
                self.residual_convs.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=1, stride=1))
            else:
                self.residual_convs.append(None)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            identity = x
            x = layer(x)
            if i > 0:
                if self.residual_convs[i] is not None:
                    identity = self.residual_convs[i](identity)
                x = x + identity
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x