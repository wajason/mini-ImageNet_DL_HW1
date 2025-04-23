import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-Head Self-Attention 層（Transformer）
class MultiHeadSelfAttention(nn.Module):
    # 初始化多頭自注意力模組、頭數設為 4 
    def __init__(self, in_channels, num_heads=1):  # 減少頭數到 1
        super(MultiHeadSelfAttention, self).__init__()
        # 每個頭處理的通道數
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        # 使用 1x1 卷積層生成 Query (Q)、Key (K)、Value (V) 和輸出投影，通道數不變。
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = (self.head_dim) ** -0.5
        self.dropout = nn.Dropout(0.2)

    # 定義前向傳播函數
    def forward(self, x):
        # x: (batch_size, in_channels, height, width)
        batch_size, C, H, W = x.size()
        # 攤平成一維序列
        N = H * W

        # 將 x 轉換為 Q、K、V，並調整形狀
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # 計算注意力分數
        energy = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # 將分數正規劃為注意力權重
        attention = F.softmax(energy, dim=-1)
        # dropout正則化
        attention = self.dropout(attention)

        # 計算加權值
        out = torch.matmul(attention, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, self.in_channels, H, W)
        out = self.out(out)
        return out

# DropBlock 正規化（針對卷積層）
class DropBlock(nn.Module):
    # 初始化 DropBlock 模組，預設7*7
    def __init__(self, block_size=7, drop_prob=0.2):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    # 定義前向傳播函數
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        batch_size, channels, height, width = x.size()

        # 計算每個區塊的丟棄概率gamma
        gamma = self.drop_prob / (self.block_size ** 2) * (height * width) / ((height - self.block_size + 1) * (width - self.block_size + 1))
        # 使用伯努利分布隨機生成丟棄區塊的遮罩
        mask = torch.bernoulli(torch.ones(batch_size, channels, height, width, device=x.device) * gamma)

        # 將遮罩展開為區塊大小
        mask = F.max_pool2d(mask, self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask

        out = x * mask
        out = out * (mask.numel() / mask.sum())
        return out

# 新模型：WideAttnCNN（2 層卷積 + 1 層 Transformer）
class WideAttnCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(WideAttnCNN, self).__init__()
        # 第一層卷積，輸入 3 channels, 輸出 32 channels（減少通道數）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, groups=1)
        # 批量正規化
        self.bn1 = nn.BatchNorm2d(32)
        # DropBlock 正規化
        self.dropblock1 = DropBlock(block_size=7, drop_prob=0.2)
        
        # 第二層卷積，輸入 32 channels, 輸出 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=1)
        # 批量正規化
        self.bn2 = nn.BatchNorm2d(64)
        # DropBlock 正規化
        self.dropblock2 = DropBlock(block_size=7, drop_prob=0.2)

        # 多頭自注意力層，輸入 64 channels, 輸出 64 channels
        self.attn = MultiHeadSelfAttention(64, num_heads=1)
        # 層正規化（調整為動態形狀）
        self.ln = nn.LayerNorm([64])
        # 池化層
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全連接層，輸入 64 channels, 輸出 num_classes
        self.fc = nn.Linear(64, num_classes)

    # 定義前向傳播函數
    def forward(self, x):
        # 第一層卷積、BN、ReLU 和 DropBlock。
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropblock1(x)
        # 第二層卷積、BN、ReLU 和 DropBlock。
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropblock2(x)
        # 多頭自注意力層
        x = self.attn(x) + x
        # 動態獲取形狀以適配不同分辨率
        batch_size, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        x = self.ln(x)
        x = x.view(batch_size, H, W, C).permute(0, 3, 1, 2)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # 全連接層
        x = self.fc(x)
        return x

# 模型 1：WideCNN（增強版）
class WideCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(WideCNN, self).__init__()
        # 第一層卷積，輸入 3 channels, 輸出 128 channels（增加通道數）
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        # 第二層卷積，輸入 128 channels, 輸出 256 channels
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        # 殘差連接（調整通道數）
        self.conv1x1_1 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        
        # 第三層卷積，輸入 256 channels, 輸出 512 channels
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        # 殘差連接
        self.conv1x1_2 = nn.Conv2d(256, 512, kernel_size=1, padding=0)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        # 全連接層，輸入 512 channels, 輸出 num_classes
        self.fc = nn.Linear(512, num_classes)

    # 定義前向傳播函數
    def forward(self, x):
        # 第一層卷積、BN、ReLU
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 第二層卷積、BN、ReLU（殘差連接）
        identity = self.conv1x1_1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + identity
        
        # 第三層卷積、BN、ReLU（殘差連接）
        identity = self.conv1x1_2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x + identity
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # 全連接層
        x = self.fc(x)
        return x

# 模型 2：AttnCNN（原模型，降低通道數）
class AttnCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(AttnCNN, self).__init__()
        # 第一層卷積，輸入 3 channels, 輸出 16 channels（減少通道數）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # 第二層卷積，輸入 16 channels, 輸出 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 第三層卷積，輸入 32 channels, 輸出 64 channels
        self.attn = MultiHeadSelfAttention(32, num_heads=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全連接層，輸入 32 channels, 輸出 num_classes
        self.fc = nn.Linear(32, num_classes)

    # 定義前向傳播函數
    def forward(self, x):
        # 第一層卷積、ReLU 和池化
        x = F.relu(self.conv1(x))
        # 第二層卷積、ReLU 和池化
        x = F.relu(self.conv2(x))
        # 第三層卷積、ReLU 和池化
        x = self.attn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # 全連接層
        x = self.fc(x)
        return x

# 改進的 GCN 層，引入注意力機制
class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(GCNConv, self).__init__()
        # 線性層
        self.linear = nn.Linear(in_channels, out_channels)
        # 注意力機制，計算邊權重
        self.attn = nn.Linear(in_channels * 2, 1)
        # Dropout 層
        self.dropout = nn.Dropout(dropout)

    # 定義前向傳播函數
    def forward(self, x, edge_index):
        batch_size = x.size(0)
        feat_dim = x.size(1)

        # 如果 edge_index 為空（例如 batch_size=1），直接返回線性變換
        if edge_index.size(1) == 0:
            return self.linear(x)

        row, col = edge_index

        # 計算注意力權重
        edge_features = torch.cat([x[row], x[col]], dim=1)  # (num_edges, 2 * feat_dim)
        # 使用 LeakyReLU 激活函數計算
        edge_weights = F.leaky_relu(self.attn(edge_features))  # (num_edges, 1)
        # 將邊權重轉換為一維
        edge_weights = F.softmax(edge_weights, dim=0)
        # dropout 層
        edge_weights = self.dropout(edge_weights)

        # 向量化聚合
        weighted_features = x[col] * edge_weights  # (num_edges, feat_dim)
        out = torch.zeros(batch_size, feat_dim, device=x.device)
        out = out.scatter_add_(0, row.unsqueeze(-1).expand(-1, feat_dim), weighted_features)

        # 殘差連接
        out = self.linear(out) + x
        return out

# 模型 3：GCNModel（增強版）
class GCNModel(nn.Module):
    def __init__(self, in_feats=128, hid_feats=256, num_classes=50):  # 增加特徵和隱藏層維度
        super(GCNModel, self).__init__()
        # 增加 1 層卷積，提升特徵提取能力
        self.conv1 = nn.Conv2d(3, in_feats, kernel_size=3, padding=1)
        # 批量正規化
        self.bn1 = nn.BatchNorm2d(in_feats)
        # 第二層卷積，輸入 in_feats channels, 輸出 in_feats * 2 channels
        self.conv2 = nn.Conv2d(in_feats, in_feats * 2, kernel_size=3, padding=1)
        # 批量正規化
        self.bn2 = nn.BatchNorm2d(in_feats * 2)
        # 第三層卷積
        self.conv3 = nn.Conv2d(in_feats * 2, in_feats * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(in_feats * 2)
        # 池化層
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 攤平
        self.flatten = nn.Flatten()

        # GCN 層
        # 第一層 GCN，輸入 in_feats * 2 channels, 輸出 hid_feats channels
        self.gcn1 = GCNConv(in_feats * 2, hid_feats, dropout=0.1)
        # 第二層 GCN，輸入 hid_feats channels, 輸出 hid_feats channels
        self.gcn2 = GCNConv(hid_feats, hid_feats, dropout=0.1)

        # Dropout 和全連接層
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hid_feats, num_classes)

        # 儲存 edge_index 的緩存
        self.edge_index_cache = {}

    # 定義前向傳播函數
    def forward(self, x):
        batch_size = x.size(0)
        # 特徵提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.flatten(x)  # x: (batch_size, in_feats * 2)

        # 使用預計算的 edge_index（固定 k-NN 圖，k=4）
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

        # GCN 層
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def clear_cache(self):
        self.edge_index_cache.clear()

# 模型 4：BiGCNModel（增強版）
class BiGCNModel(nn.Module):
    def __init__(self, in_feats=128, hid_feats=256, num_classes=50):  # 增加特徵和隱藏層維度
        super(BiGCNModel, self).__init__()
        # 第一層卷積，輸入 3 channels, 輸出 in_feats channels
        self.conv1 = nn.Conv2d(3, in_feats, kernel_size=3, padding=1)
        # 批量正規化
        self.bn1 = nn.BatchNorm2d(in_feats)
        # 第二層卷積，輸入 in_feats channels, 輸出 in_feats * 2 channels
        self.conv2 = nn.Conv2d(in_feats, in_feats * 2, kernel_size=3, padding=1)
        # 批量正規化
        self.bn2 = nn.BatchNorm2d(in_feats * 2)
        # 第三層卷積
        self.conv3 = nn.Conv2d(in_feats * 2, in_feats * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(in_feats * 2)
        # 池化層
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 攤平
        self.flatten = nn.Flatten()

        # 第二層GCN 層
        self.td_gcn1 = GCNConv(in_feats * 2, hid_feats, dropout=0.1)
        # 第三層 GCN 層
        self.bu_gcn1 = GCNConv(in_feats * 2, hid_feats, dropout=0.1)
        self.gcn2 = GCNConv(hid_feats * 2, hid_feats, dropout=0.1)

        # Dropout 和全連接層
        self.dropout = nn.Dropout(0.1)
        # 全連接層
        self.fc = nn.Linear(hid_feats, num_classes)

        # 儲存 edge_index 的緩存
        self.edge_index_cache = {}

    # 定義前向傳播函數
    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.flatten(x)

        # 使用預計算的 edge_index（固定 k-NN 圖，k=4）
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

        td_x = F.relu(self.td_gcn1(x, edge_index))
        bu_x = F.relu(self.bu_gcn1(x, edge_index.flip(0)))
        x = torch.cat((td_x, bu_x), dim=1)
        x = F.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)
        x = self.fc(x)
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