import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from torchvision.models import ResNet18_Weights  # 導入 ResNet18_Weights 用於加載預訓練權重

# 自定義 mini-ImageNet 數據集類
class MiniImageNetDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        # 讀取 txt 檔案，包含圖像路徑和標籤
        self.data = pd.read_csv(txt_file, sep=" ", header=None, names=["filepath", "label"])
        self.root_dir = root_dir
        self.transform = transform
        # 移除圖像路徑中的 'images/' 前綴，避免路徑重複
        self.data['filepath'] = self.data['filepath'].str.replace('images/', '', 1)
        # 創建標籤到整數的映射
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}
        self.data['label'] = self.data['label'].map(self.label_map)

    def __len__(self):
        # 返回數據集的大小
        return len(self.data)

    def __getitem__(self, idx):
        # 加載圖像
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx, 1]

        # 應用數據增強或轉換
        if self.transform:
            image = self.transform(image)

        return image, label

# 定義訓練數據的轉換（包含數據增強）
transform_train = transforms.Compose([
    transforms.Resize((84, 84)),  # 將圖像調整為 84x84（mini-ImageNet 常用大小）
    transforms.RandomHorizontalFlip(),  # 隨機水平翻轉，用於數據增強
    transforms.RandomRotation(10),  # 隨機旋轉 10 度，用於數據增強
    transforms.ToTensor(),  # 轉換為張量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和標準差進行標準化
])

# 定義驗證和測試數據的轉換（不包含數據增強）
transform_val_test = transforms.Compose([
    transforms.Resize((84, 84)),  # 將圖像調整為 84x84
    transforms.ToTensor(),  # 轉換為張量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和標準差進行標準化
])

# 加載數據集
root_dir = "/home/wajason99/mini-ImageNet_DL_HW1/images"  # 圖像檔案的根目錄路徑
train_dataset = MiniImageNetDataset(txt_file="/home/wajason99/mini-ImageNet_DL_HW1/train.txt", root_dir=root_dir, transform=transform_train)
val_dataset = MiniImageNetDataset(txt_file="/home/wajason99/mini-ImageNet_DL_HW1/val.txt", root_dir=root_dir, transform=transform_val_test)
test_dataset = MiniImageNetDataset(txt_file="/home/wajason99/mini-ImageNet_DL_HW1/test.txt", root_dir=root_dir, transform=transform_val_test)

# 創建數據加載器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# 加載預訓練的 ResNet-18 模型
model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 使用 weights 參數加載預訓練權重
# 修改最後的全連接層以匹配 mini-ImageNet 的類別數（應為 100 類）
num_classes = len(train_dataset.label_map)  # 計算類別數量
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 將模型移動到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 訓練迴圈
num_epochs = 50
best_val_acc = 0.0
for epoch in range(num_epochs):
    # 訓練階段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total
    train_loss = train_loss / len(train_loader)

    # 驗證階段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.2f}%")
    print(f"驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.2f}%")

    # 根據驗證準確率保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

# 加載最佳模型並在測試集上評估
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * test_correct / test_total
test_loss = test_loss / len(test_loader)

print("\n最終測試表現:")
print(f"測試損失: {test_loss:.4f}, 測試準確率: {test_acc:.2f}%")