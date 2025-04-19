import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, K=2):
        super(DynamicConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, int) else tuple(stride)
        self.K = K

        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size, stride=stride, padding=kernel_size//2)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, K),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels, f"Input channels mismatch: expected {self.in_channels}, got {in_channels}"

        attn_weights = self.attention(x)
        out = self.conv(x)
        out = out.view(batch_size, self.out_channels, self.K, height, width)
        attn_weights = attn_weights.view(batch_size, 1, self.K, 1, 1)
        out = (out * attn_weights).sum(dim=2)
        return out

class TaskAModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=50, num_layers=5):
        super(TaskAModel, self).__init__()
        self.layers = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        channels = [in_channels, 32, 64, 128, 256, 512][:num_layers+1]

        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    DynamicConv(channels[i], channels[i+1], kernel_size=3, stride=1),
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
        self.num_layers = num_layers

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