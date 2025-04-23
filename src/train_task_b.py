import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from model_task_b import WideCNN, AttnCNN, WideAttnCNN, GCNModel, BiGCNModel, WideCNN_2Layer, AttnCNN_NoAttn
from data_loader import train_loader, val_loader, test_loader, mixup_data
import psutil
import pynvml
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader # 數據集和數據加載器

# 設置儲存路徑
curve_plot_dir = "/home/wajason99/mini-ImageNet_DL_HW1/curve_plot_B"
os.makedirs(curve_plot_dir, exist_ok=True)

# 限制 PyTorch 線程數和 GPU 設置
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 GPU 監控
try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except:
    gpu_handle = None

# 手動計算參數量
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Manually calculated parameters for {model.__class__.__name__}: {total_params / 1e6:.2f} M")
    return total_params / 1e6

# 計算 FLOPS 和參數量
def get_model_stats(model, input_size=(3, 224, 224)):  # 確保圖像大小為 224x224
    from thop import profile
    model.eval()
    input_tensor = torch.randn(1, *input_size).to(device)
    try:
        flops, params = profile(model, inputs=(input_tensor,), verbose=True)
        print(f"thop.profile succeeded for {model.__class__.__name__}: FLOPS={flops / 1e9:.2f} GFLOPS, Params={params / 1e6:.2f} M")
        return flops / 1e9, params / 1e6
    except Exception as e:
        print(f"Error calculating stats for model {model.__class__.__name__}: {e}")
        params = count_parameters(model)
        return 0, params

def print_resource_usage():
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=1)
    memory_info = process.memory_info()
    print(f"CPU Usage: {cpu_percent:.2f}%")
    print(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")

    if gpu_handle:
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        print(f"GPU Memory Usage: {gpu_mem.used / 1024 / 1024:.2f} MB / {gpu_mem.total / 1024 / 1024:.2f} MB")
        print(f"GPU Utilization: {gpu_util.gpu}%")

# Warmup 學習率調度器
class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            cosine_epoch = self.last_epoch - self.warmup_epochs
            cosine_total = self.total_epochs - self.warmup_epochs
            lr = [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + np.cos(np.pi * cosine_epoch / cosine_total)) / 2
                for base_lr in self.base_lrs
            ]
        return lr

def train_and_evaluate(model, model_name="Model", epochs=5, batch_size=128):
    torch.cuda.empty_cache()
    
    # 動態設置 DataLoader 的 batch_size
    train_loader.batch_size = batch_size
    val_loader.batch_size = batch_size
    test_loader.batch_size = batch_size
    
    train_loader_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader_loader = DataLoader(val_loader.dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader_loader = DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, total_epochs=epochs, eta_min=0)
    
    scaler = torch.amp.GradScaler('cuda')

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    best_model_state = None

    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        # 清除 edge_index 緩存
        if hasattr(model, 'clear_cache'):
            model.clear_cache()

        model.train()
        loss_train, acc_train = 0.0, 0.0
        optimizer.zero_grad()
        for i, (images, labels) in enumerate(train_loader_loader):
            print(f"[{model_name}] Processing batch {i+1}/{len(train_loader_loader)}")
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_train += loss.item()
            preds = outputs.argmax(dim=1)
            acc_train += (preds == labels).float().mean().item()
            print(f"[{model_name}] Batch {i+1} loss: {loss.item():.4f}")

        loss_train /= len(train_loader_loader)
        acc_train /= len(train_loader_loader)
        train_losses.append(loss_train)
        train_accuracies.append(acc_train)

        model.eval()
        loss_val, acc_val = 0.0, 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader_loader, desc="Validating", leave=False):
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss_val += loss.item()
                preds = outputs.argmax(dim=1)
                acc_val += (preds == labels).float().mean().item()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        loss_val /= len(val_loader_loader)
        acc_val /= len(val_loader_loader)
        val_losses.append(loss_val)
        val_accuracies.append(acc_val)

        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted', zero_division=0)

        scheduler.step()
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_model_state = model.state_dict()

        print(f"[{model_name}] Epoch {epoch+1}/{epochs}: Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.4f}, "
              f"Val Loss: {loss_val:.4f}, Val Acc: {acc_val:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        print_resource_usage()

    torch.save(best_model_state, os.path.join(curve_plot_dir, f"best_model_{model_name}.pt"))

    model.eval()
    test_acc, test_loss = 0.0, 0.0
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            test_loss += loss.item()
            preds = outputs.argmax(dim=1)
            test_acc += (preds == labels).float().mean().item()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader_loader)
    test_acc /= len(test_loader_loader)
    
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted', zero_division=0)
    
    print(f"[{model_name}] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
          f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, "
          f"Best Val Acc: {best_val_acc:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({model_name})")
    plt.legend()
    plt.savefig(os.path.join(curve_plot_dir, f"loss_curve_{model_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label="Train Acc")
    plt.plot(range(1, epochs + 1), val_accuracies, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve ({model_name})")
    plt.legend()
    plt.savefig(os.path.join(curve_plot_dir, f"accuracy_curve_{model_name}.png"))
    plt.close()

    # 釋放記憶體
    del model
    torch.cuda.empty_cache()

    return test_acc, best_val_acc, test_precision, test_recall, test_f1

if __name__ == "__main__":

    # 訓練 ResNet34（已完成，註解掉）
    """
    resnet = resnet34(weights=None, num_classes=50).to(device)
    resnet_flops, resnet_params = get_model_stats(resnet)
    resnet_acc, resnet_val_acc, resnet_precision, resnet_recall, resnet_f1 = train_and_evaluate(resnet, "ResNet34", epochs=5)
    """

    resnet_acc = 0.4355
    resnet_val_acc = 0.5059
    resnet_precision = 0.5230
    resnet_recall = 0.4956
    resnet_f1 = 0.4704

    # 為 WideCNN、GCNModel 和 BiGCNModel 使用更大的 batch_size
    models_high_batch = [
        ("WideCNN", WideCNN(num_classes=50).to(device)),
        ("GCNModel", GCNModel(in_feats=128, hid_feats=256, num_classes=50).to(device)),
        ("BiGCNModel", BiGCNModel(in_feats=128, hid_feats=256, num_classes=50).to(device))
    ]

    # 為 AttnCNN 和 WideAttnCNN 使用較小的 batch_size
    models_low_batch = [
        ("AttnCNN", AttnCNN(num_classes=50).to(device)),
        ("WideAttnCNN", WideAttnCNN(num_classes=50).to(device))
    ]

    results = {
        "WideCNN": (widecnn_acc, widecnn_val_acc, widecnn_precision, widecnn_recall, widecnn_f1, widecnn_flops, widecnn_params)
    }

    # 訓練 WideCNN、GCNModel 和 BiGCNModel
    for model_name, model in models_high_batch:
        torch.cuda.empty_cache()
        flops, params = get_model_stats(model)
        print(f"\n[{model_name}] FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")
        test_acc, val_acc, test_precision, test_recall, test_f1 = train_and_evaluate(model, model_name, epochs=5, batch_size=128)
        results[model_name] = (test_acc, val_acc, test_precision, test_recall, test_f1, flops, params)

    # 訓練 AttnCNN 和 WideAttnCNN
    for model_name, model in models_low_batch:
        torch.cuda.empty_cache()
        flops, params = get_model_stats(model)
        print(f"\n[{model_name}] FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")
        test_acc, val_acc, test_precision, test_recall, test_f1 = train_and_evaluate(model, model_name, epochs=5, batch_size=2)
        results[model_name] = (test_acc, val_acc, test_precision, test_recall, test_f1, flops, params)

    ablation_models = [
        ("WideCNN_2Layer", WideCNN_2Layer(num_classes=50).to(device)),
        ("AttnCNN_NoAttn", AttnCNN_NoAttn(num_classes=50).to(device)),
    ]

    print("\nAblation Study:")
    for model_name, model in ablation_models:
        torch.cuda.empty_cache()
        flops, params = get_model_stats(model)
        print(f"\n[{model_name}] FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")
        test_acc, val_acc, test_precision, test_recall, test_f1 = train_and_evaluate(model, model_name, epochs=5, batch_size=128)
        results[model_name] = (test_acc, val_acc, test_precision, test_recall, test_f1, flops, params)

    print("\nFinal Performance Comparison (Validation Set):")
    print(f"ResNet34 Val Acc: {resnet_val_acc:.4f}")
    for model_name, (test_acc, val_acc, _, _, _, _, _) in results.items():
        print(f"{model_name} Val Acc: {val_acc:.4f}, Relative to ResNet34: {(val_acc / resnet_val_acc) * 100:.2f}%")

    print("\nFinal Performance Comparison (Test Set):")
    print(f"ResNet34 Test Acc: {resnet_acc:.4f}, Precision: {resnet_precision:.4f}, Recall: {resnet_recall:.4f}, F1: {resnet_f1:.4f}")
    for model_name, (test_acc, val_acc, test_precision, test_recall, test_f1, flops, params) in results.items():
        print(f"{model_name} Test Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, "
              f"Relative to ResNet34: {(test_acc / resnet_acc) * 100:.2f}%, "
              f"FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")

    if gpu_handle:
        pynvml.nvmlShutdown()