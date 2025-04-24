import os
import torch
import pandas as pd
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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

# 定義數據集類
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

# 資料增強
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4799, 0.4597, 0.3874], std=[0.2098, 0.2032, 0.1980]),
    Cutout(n_holes=1, length=32)
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

# 數據加載（batch_size 將在 train_task_b.py 中動態設置）
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)