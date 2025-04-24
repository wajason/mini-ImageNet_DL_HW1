import os
import pandas as pd
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torch
from tqdm import tqdm

# 定義 Cutout 數據增強
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask).to(img.device)
        mask = mask.expand_as(img)
        img = img * mask

        return img

# 定義 Mixup 數據增強
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class ChannelSelector:
    def __init__(self, channels='RGB'):
        self.channels = channels
        self.channel_indices = {'R': 0, 'G': 1, 'B': 2}
        self.selected_indices = [self.channel_indices[c] for c in channels]

    def __call__(self, img):
        return img[self.selected_indices]

class MiniImageNetDataset(Dataset):
    def __init__(self, txt_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = []
        with open(txt_file, 'r') as f:
            for line in f:
                image_path, label = line.strip().split()
                self.data.append((image_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 數據加載器
sizes = [224]
datasets = {}
loaders = {}

# ImageNet 標準均值和標準差（RGB）
base_mean = [0.485, 0.456, 0.406]
base_std = [0.229, 0.224, 0.225]

for size in sizes:
    # 訓練和驗證的正規化（僅 RGB）
    normalize = transforms.Normalize(mean=base_mean, std=base_std)

    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize
    ])

    # 測試的通道組合和對應的正規化
    channel_combinations = ['RGB', 'RG', 'GB', 'RB', 'R', 'G', 'B']
    test_transforms = []
    for channels in channel_combinations:
        # 根據通道組合選擇均值和標準差
        indices = [0, 1, 2] if channels == 'RGB' else \
                  [0, 1] if channels == 'RG' else \
                  [1, 2] if channels == 'GB' else \
                  [0, 2] if channels == 'RB' else \
                  [0] if channels == 'R' else \
                  [1] if channels == 'G' else \
                  [2]  # B
        mean = [base_mean[i] for i in indices]
        std = [base_std[i] for i in indices]
        test_transforms.append(
            transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                ChannelSelector(channels),
                transforms.Normalize(mean=mean, std=std)
            ])
        )

    train_dataset = MiniImageNetDataset("data/train.txt", "data", transform=train_transform)
    val_dataset = MiniImageNetDataset("data/val.txt", "data", transform=val_transform)
    test_dataset = MiniImageNetDataset("data/test.txt", "data")

    datasets[size] = {
        'train': train_dataset,
        'val': val_dataset,
        'test': [MiniImageNetDataset("data/test.txt", "data", transform=t) for t in test_transforms]
    }

    loaders[size] = {
        'train': DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=15, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=96, shuffle=False, num_workers=15, pin_memory=True),
        'test': [DataLoader(test_ds, batch_size=96, shuffle=False, num_workers=15, pin_memory=True) for test_ds in datasets[size]['test']]
    }