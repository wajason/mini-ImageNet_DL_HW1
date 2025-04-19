import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, K=4):
        super(DynamicConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, int) else tuple(stride)
        self.K = K  # 候選特徵加權數量

        # 標準卷積層，處理整個批次
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size, stride=stride, padding=kernel_size//2)

        # 注意力網路：為每個樣本生成 K 個加權係數
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, K),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels, f"Input channels mismatch: expected {self.in_channels}, got {in_channels}"

        # 計算注意力權重（每個樣本獨立）
        attn_weights = self.attention(x)  # (batch_size, K)

        # 標準卷積，輸出 K 組特徵圖
        out = self.conv(x)  # (batch_size, out_channels * K, height, width)

        # 將輸出拆分為 K 組特徵圖
        out = out.view(batch_size, self.out_channels, self.K, height, width)  # (batch_size, out_channels, K, height, width)

        # 應用注意力權重，對 K 組特徵圖加權求和
        attn_weights = attn_weights.view(batch_size, 1, self.K, 1, 1)  # (batch_size, 1, K, 1, 1)
        out = (out * attn_weights).sum(dim=2)  # (batch_size, out_channels, height, width)

        return out

class TaskAModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=50, num_layers=5):
        super(TaskAModel, self).__init__()
        self.layers = nn.ModuleList()
        channels = [in_channels, 32, 64, 128, 256, 512][:num_layers+1]

        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    DynamicConv(channels[i], channels[i+1], kernel_size=3, stride=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU()
                )
            )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x