import os              # 處理檔案和目錄操作
import pandas as pd    # 數據處理和分析
import numpy as np     # 數值計算
import random          # 隨機數生成
from PIL import Image  # 圖像處理
from torch.utils.data import Dataset, DataLoader # 數據集和數據加載器
import torchvision.transforms as transforms      # 圖像轉換

# 定義一個數據集類，用於加載 mini-ImageNet 數據集
class MiniImageNetDataset(Dataset):
    # 初始化函數
    def __init__(self, txt_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = []
        with open(txt_file, 'r') as f:
            for line in f:
                image_path, label = line.strip().split()
                self.data.append((image_path, int(label)))

    # 獲取數據集的大小
    def __len__(self):
        return len(self.data)

    # 獲取指定索引的數據
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 資料增強
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),     # 調整大小為 224x224
    transforms.RandomHorizontalFlip(), # 隨機水平翻轉
    transforms.RandomRotation(15),     # 隨機旋轉 15 度
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # 隨機顏色抖動
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 隨機平移
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4799, 0.4597, 0.3874], std=[0.2098, 0.2032, 0.1980]) # 正規化
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4799, 0.4597, 0.3874], std=[0.2098, 0.2032, 0.1980])
])

# 檔案路徑
train_dataset = MiniImageNetDataset("data/train.txt", "data", transform=transform_train)
val_dataset = MiniImageNetDataset("data/val.txt", "data", transform=transform_val)
test_dataset = MiniImageNetDataset("data/test.txt", "data", transform=transform_val)

# 數據加載
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)