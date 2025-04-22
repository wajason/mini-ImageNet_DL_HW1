# src/model_task_b.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

# Multi-Head Self-Attention 層（Transformer 風格）
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        N = H * W

        # 計算 Q, K, V
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, heads, N, head_dim)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)    # (B, heads, N, head_dim)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, heads, N, head_dim)

        # 注意力計算
        energy = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, N, N)
        attention = F.softmax(energy, dim=-1)  # (B, heads, N, N)
        attention = self.dropout(attention)

        # 輸出
        out = torch.matmul(attention, v)  # (B, heads, N, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, self.in_channels, H, W)  # (B, C, H, W)
        out = self.out(out)
        return out

# DropBlock 正規化（針對卷積層）
class DropBlock(nn.Module):
    def __init__(self, block_size=7, drop_prob=0.1):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        batch_size, channels, height, width = x.size()

        # 計算每個位置保留的概率
        gamma = self.drop_prob / (self.block_size ** 2) * (height * width) / ((height - self.block_size + 1) * (width - self.block_size + 1))
        mask = torch.bernoulli(torch.ones(batch_size, channels, height, width, device=x.device) * gamma)

        # 將 mask 中為 1 的位置周圍 block_size x block_size 區域設為 0
        mask = F.max_pool2d(mask, self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask

        # 應用 mask 並正規化
        out = x * mask
        out = out * (mask.numel() / mask.sum())
        return out

# 新模型：WideAttnCNN（2 層卷積 + 1 層 Transformer）
class WideAttnCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(WideAttnCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, groups=3)  # Grouped Convolution
        self.bn1 = nn.BatchNorm2d(128)
        self.dropblock1 = DropBlock(block_size=7, drop_prob=0.1)
        
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, groups=3)
        self.bn2 = nn.BatchNorm2d(256)
        self.dropblock2 = DropBlock(block_size=7, drop_prob=0.1)

        self.attn = MultiHeadSelfAttention(256, num_heads=8)
        self.ln = nn.LayerNorm([256, 224, 224])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropblock1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropblock2(x)
        x = self.attn(x) + x  # 殘差連接
        x = self.ln(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 簡單的 GCN 層
class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, batch_size, num_nodes):
        row, col = edge_index
        out = scatter_mean(x[col], row, dim=0, dim_size=batch_size * num_nodes)
        out = self.linear(out)
        return out

# 模型 1：WideCNN（原模型）
class WideCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(WideCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 模型 2：AttnCNN（原模型）
class AttnCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(AttnCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.attn = MultiHeadSelfAttention(256, num_heads=8)  # 更新為 MHSA
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.attn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 模型 3：GCNModel
class GCNModel(nn.Module):
    def __init__(self, in_feats=64, hid_feats=128, num_classes=50):
        super(GCNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, in_feats, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.gcn1 = GCNConv(in_feats, hid_feats)
        self.gcn2 = GCNConv(hid_feats, hid_feats)
        self.fc = nn.Linear(hid_feats, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        num_nodes = 16
        x = x.view(batch_size * num_nodes, -1)

        edge_index_base = torch.combinations(torch.arange(num_nodes), r=2).t().to(x.device)
        num_edges_per_batch = edge_index_base.size(1)
        edge_index = edge_index_base.repeat(1, batch_size)
        offset = torch.arange(batch_size, device=x.device) * num_nodes
        offset = offset.repeat_interleave(num_edges_per_batch).view(1, -1)
        edge_index += offset

        x = F.relu(self.gcn1(x, edge_index, batch_size, num_nodes))
        x = F.relu(self.gcn2(x, edge_index, batch_size, num_nodes))
        batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes).to(x.device)
        x = scatter_mean(x, batch, dim=0)
        x = self.fc(x)
        return x

# 模型 4：BiGCNModel
class BiGCNModel(nn.Module):
    def __init__(self, in_feats=64, hid_feats=128, num_classes=50):
        super(BiGCNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, in_feats, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.td_gcn1 = GCNConv(in_feats, hid_feats)
        self.bu_gcn1 = GCNConv(in_feats, hid_feats)
        self.gcn2 = GCNConv(hid_feats * 2, hid_feats)
        self.fc = nn.Linear(hid_feats, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        num_nodes = 16
        x = x.view(batch_size * num_nodes, -1)

        edge_index_base = torch.combinations(torch.arange(num_nodes), r=2).t().to(x.device)
        num_edges_per_batch = edge_index_base.size(1)
        edge_index = edge_index_base.repeat(1, batch_size)
        offset = torch.arange(batch_size, device=x.device) * num_nodes
        offset = offset.repeat_interleave(num_edges_per_batch).view(1, -1)
        edge_index += offset

        td_x = F.relu(self.td_gcn1(x, edge_index, batch_size, num_nodes))
        bu_x = F.relu(self.bu_gcn1(x, edge_index.flip(0), batch_size, num_nodes))
        x = torch.cat((td_x, bu_x), dim=1)
        x = F.relu(self.gcn2(x, edge_index, batch_size, num_nodes))
        batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes).to(x.device)
        x = scatter_mean(x, batch, dim=0)
        x = self.fc(x)
        return x

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
        return x