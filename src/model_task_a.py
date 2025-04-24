import torch
import torch.nn as nn
import torch.nn.functional as F

class attention2d(nn.Module):
    def __init__(self, in_planes, ratios=0.25, K=2, tau=30.0):
        super(attention2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_planes = max(int(in_planes * ratios), 1) if in_planes != 3 else K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.tau = tau
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.tau, dim=1)

class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, ratio=0.25, stride=1, padding=1, K=2, tau=30.0):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % 1 == 0  # groups=1
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, tau)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes, kernel_size, kernel_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(K, out_planes))
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def set_tau(self, tau):
        self.attention.set_tau(tau)

    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size * self.out_planes, self.in_planes, self.kernel_size, self.kernel_size)
        aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
        output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, groups=batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=50):
        super(ConvNet, self).__init__()
        self.layers = nn.ModuleList()
        channels = [in_channels, 32, 64, 128, 256]
        
        for i in range(len(channels) - 1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3)
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

class DynamicConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=50, K=2):
        super(DynamicConvNet, self).__init__()
        self.K = K
        self.layers = nn.ModuleList()
        channels = [in_channels, 32, 64, 128, 256]
        
        for i in range(len(channels) - 1):
            self.layers.append(nn.Sequential(
                Dynamic_conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1, K=K),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def set_tau(self, tau):
        for layer in self.layers:
            layer[0].set_tau(tau)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DynamicConvNet_3Layer(nn.Module):
    def __init__(self, in_channels=3, num_classes=50, K=2):
        super(DynamicConvNet_3Layer, self).__init__()
        self.K = K
        self.layers = nn.ModuleList()
        channels = [in_channels, 32, 64, 128]
        
        for i in range(len(channels) - 1):
            self.layers.append(nn.Sequential(
                Dynamic_conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1, K=K),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def set_tau(self, tau):
        for layer in self.layers:
            layer[0].set_tau(tau)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DynamicConvNet_2Layer(nn.Module):
    def __init__(self, in_channels=3, num_classes=50, K=2):
        super(DynamicConvNet_2Layer, self).__init__()
        self.K = K
        self.layers = nn.ModuleList()
        channels = [in_channels, 32, 64]
        
        for i in range(len(channels) - 1):
            self.layers.append(nn.Sequential(
                Dynamic_conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1, K=K),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def set_tau(self, tau):
        for layer in self.layers:
            layer[0].set_tau(tau)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x