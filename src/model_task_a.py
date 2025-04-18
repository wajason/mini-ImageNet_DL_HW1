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
        self.K = K  # 候選卷積核數量

        # 候選卷積核參數
        self.weight = nn.Parameter(
            torch.randn(K, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(K, out_channels))

        # 注意力網路：為每個樣本生成獨立的權重
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
        assert in_channels == self.in_channels, "Input channels mismatch"

        # 計算注意力權重（每個樣本獨立）
        attn_weights = self.attention(x)  # (batch_size, K)

        # 動態生成卷積核
        weights = self.weight.view(self.K, -1)  # (K, out_channels * in_channels * kernel_size * kernel_size)
        attn_weights_for_weights = attn_weights.view(batch_size, self.K, 1)  # (batch_size, K, 1)
        dynamic_weight = (attn_weights_for_weights * weights).sum(dim=1)  # (batch_size, out_channels * in_channels * kernel_size * kernel_size)
        dynamic_weight = dynamic_weight.view(batch_size, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        # 動態生成偏置
        attn_weights_for_bias = attn_weights.view(batch_size, self.K, 1)  # (batch_size, K, 1)
        dynamic_bias = (attn_weights_for_bias * self.bias.unsqueeze(0)).sum(dim=1)  # (batch_size, out_channels)

        # 對每個樣本進行卷積
        out = torch.zeros(batch_size, self.out_channels, height, width, device=x.device)
        for i in range(batch_size):
            out[i] = F.conv2d(
                x[i:i+1],
                dynamic_weight[i],
                bias=dynamic_bias[i],
                stride=self.stride,
                padding=self.kernel_size//2
            )
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