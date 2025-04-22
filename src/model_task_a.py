import torch
import torch.nn as nn
import torch.nn.functional as F

# 改進的 DynamicConv
class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, K=4, tau=30.0):
        super(DynamicConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, int) else tuple(stride)
        self.K = K
        self.tau = tau  # 溫度參數

        # 獨立的 K 個卷積核
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
            for _ in range(K)
        ])

        # 增強的注意力模組
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),  # 更大的中間層
            nn.ReLU(),
            nn.Linear(256, K),
            nn.Softmax(dim=1)
        )

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels, f"Input channels mismatch: expected {self.in_channels}, got {in_channels}"

        # 計算注意力權重
        attn_weights = self.attention(x)  # (batch_size, K)
        attn_weights = attn_weights / self.tau  # 應用溫度參數
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_weights = attn_weights.view(batch_size, self.K, 1, 1, 1)

        # 計算 K 個卷積核的輸出並加權
        out = 0
        for k in range(self.K):
            conv_out = self.convs[k](x)  # (batch_size, out_channels, height, width)
            conv_out = conv_out.unsqueeze(1)  # (batch_size, 1, out_channels, height, width)
            out += conv_out * attn_weights[:, k]

        return out

# 普通卷積網絡（10 層）
class ConvNet(nn.Module):
    def __init__(self, num_classes=50):
        super(ConvNet, self).__init__()
        self.layers = nn.ModuleList()
        channels = [3, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512]
        
        for i in range(len(channels) - 1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# DynamicConv 網絡（10 層）
class DynamicConvNet(nn.Module):
    def __init__(self, num_classes=50, K=4):
        super(DynamicConvNet, self).__init__()
        self.K = K
        self.layers = nn.ModuleList()
        channels = [3, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512]
        
        for i in range(len(channels) - 1):
            self.layers.append(nn.Sequential(
                DynamicConv(channels[i], channels[i+1], kernel_size=3, padding=1, K=K),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def set_tau(self, tau):
        for layer in self.layers:
            layer[0].set_tau(tau)  # 更新 DynamicConv 的 tau

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x