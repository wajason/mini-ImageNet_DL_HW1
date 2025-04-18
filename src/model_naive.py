import torch
import torch.nn as nn

class NaiveModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=50, num_layers=5):
        super(NaiveModel, self).__init__()
        self.layers = nn.ModuleList()
        channels = [in_channels, 32, 64, 128, 256, 512][:num_layers+1]

        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
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