import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Multi-Head Self-Attention 層（Transformer）
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=1):
        super(MultiHeadSelfAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = (self.head_dim) ** -0.5
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        N = H * W

        q = self.query(x).view(batch_size, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        energy = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, self.in_channels, H, W)
        out = self.out(out)
        torch.cuda.empty_cache()
        return out

# DropBlock 正規化（針對卷積層）
class DropBlock(nn.Module):
    def __init__(self, block_size=7, drop_prob=0.2):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        batch_size, channels, height, width = x.size()

        gamma = self.drop_prob / (self.block_size ** 2) * (height * width) / ((height - self.block_size + 1) * (width - self.block_size + 1))
        mask = torch.bernoulli(torch.ones(batch_size, channels, height, width, device=x.device) * gamma)

        mask = F.max_pool2d(mask, self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask

        out = x * mask
        out = out * (mask.numel() / mask.sum())
        torch.cuda.empty_cache()
        return out

# WideAttnCNN（2 層卷積 + 1 層 Transformer）
class WideAttnCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(WideAttnCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, groups=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropblock1 = DropBlock(block_size=7, drop_prob=0.2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropblock2 = DropBlock(block_size=7, drop_prob=0.2)

        self.attn = MultiHeadSelfAttention(64, num_heads=1)
        self.ln = nn.LayerNorm([64])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropblock1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropblock2(x)
        x = self.attn(x) + x
        batch_size, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        x = self.ln(x)
        x = x.view(batch_size, H, W, C).permute(0, 3, 1, 2)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        torch.cuda.empty_cache()
        return x

# WideCNN（增強版）
class WideCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(WideCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv1x1_1 = nn.Conv2d(256, 512, kernel_size=1, padding=0)
        
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv1x1_2 = nn.Conv2d(512, 1024, kernel_size=1, padding=0)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        identity = self.conv1x1_1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + identity
        
        identity = self.conv1x1_2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x + identity
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        torch.cuda.empty_cache()
        return x

# AttnCNN（修正版）
class AttnCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(AttnCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.attn = MultiHeadSelfAttention(32, num_heads=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.attn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        torch.cuda.empty_cache()
        return x

# 改進的 GCN 層，引入注意力機制
class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        batch_size = x.size(0)
        feat_dim = x.size(1)

        if edge_index.size(1) == 0:
            return self.linear(x)

        row, col = edge_index
        # 計算餘弦相似度作為邊權重
        x_norm = F.normalize(x, p=2, dim=1)  # 正規化特徵
        edge_weights = torch.sum(x_norm[row] * x_norm[col], dim=1, keepdim=True)  # 餘弦相似度
        edge_weights = F.softmax(edge_weights, dim=0)  # 正規化權重
        edge_weights = self.dropout(edge_weights)

        # 聚合特徵
        weighted_features = x[col] * edge_weights
        aggr_features = torch.zeros(batch_size, feat_dim, device=x.device)
        aggr_features = aggr_features.scatter_add_(0, row.unsqueeze(-1).expand(-1, feat_dim), weighted_features)

        out = self.linear(aggr_features) + x  # 殘差連接
        out = self.dropout(out)
        torch.cuda.empty_cache()
        return out

# GCNModel（增強版）
class GCNModel(nn.Module):
    def __init__(self, in_feats=128, hid_feats=256, num_classes=50):
        super(GCNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, in_feats, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_feats)
        self.conv2 = nn.Conv2d(in_feats, in_feats * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_feats * 2)
        self.conv3 = nn.Conv2d(in_feats * 2, in_feats * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(in_feats * 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.gcn1 = GCNConv(in_feats * 2, hid_feats, dropout=0.1)
        self.gcn2 = GCNConv(hid_feats, hid_feats, dropout=0.1)

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hid_feats, num_classes)

        self.edge_index_cache = {}

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.flatten(x)

        if batch_size not in self.edge_index_cache:
            if batch_size == 1:
                edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
            else:
                k = min(4, batch_size - 1)
                row, col = [], []
                for i in range(batch_size):
                    for j in range(1, k + 1):
                        target = (i + j) % batch_size
                        if i != target:
                            row.append(i)
                            col.append(target)
                edge_index = torch.tensor([row, col], dtype=torch.long, device=x.device)
                self.edge_index_cache[batch_size] = edge_index
        else:
            edge_index = self.edge_index_cache[batch_size]

        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)
        x = self.fc(x)
        torch.cuda.empty_cache()
        return x

    def clear_cache(self):
        self.edge_index_cache.clear()

# BiGCNModel（增強版）
class BiGCNModel(nn.Module):
    def __init__(self, in_feats=256, hid_feats=512, num_classes=50):
        super(BiGCNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, in_feats, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_feats)
        self.conv2 = nn.Conv2d(in_feats, in_feats * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_feats * 2)
        self.conv3 = nn.Conv2d(in_feats * 2, in_feats * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(in_feats * 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.td_gcn1 = GCNConv(in_feats * 2, hid_feats, dropout=0.05)
        self.bu_gcn1 = GCNConv(in_feats * 2, hid_feats, dropout=0.05)
        self.gcn2 = GCNConv(hid_feats * 2, hid_feats * 2, dropout=0.05)

        self.dropout = nn.Dropout(0.05)
        self.fc = nn.Linear(hid_feats * 2, num_classes)

        self.edge_index_cache = {}

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.flatten(x)

        if batch_size not in self.edge_index_cache:
            if batch_size == 1:
                edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
            else:
                k = min(8, batch_size - 1)
                row, col = [], []
                for i in range(batch_size):
                    for j in range(1, k + 1):
                        target = (i + j) % batch_size
                        if i != target:
                            row.append(i)
                            col.append(target)
                edge_index = torch.tensor([row, col], dtype=torch.long, device=x.device)
                self.edge_index_cache[batch_size] = edge_index
        else:
            edge_index = self.edge_index_cache[batch_size]

        td_x = F.relu(self.td_gcn1(x, edge_index))
        bu_x = F.relu(self.bu_gcn1(x, edge_index.flip(0)))
        x = torch.cat((td_x, bu_x), dim=1)
        x = F.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)
        x = self.fc(x)
        torch.cuda.empty_cache()
        return x

    def clear_cache(self):
        self.edge_index_cache.clear()

# 消融實驗模型：WideCNN_2Layer
class WideCNN_2Layer(nn.Module):
    def __init__(self, num_classes=50):
        super(WideCNN_2Layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        torch.cuda.empty_cache()
        return x

# 消融實驗模型：AttnCNN_NoAttn
class AttnCNN_NoAttn(nn.Module):
    def __init__(self, num_classes=50):
        super(AttnCNN_NoAttn, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        torch.cuda.empty_cache()
        return x