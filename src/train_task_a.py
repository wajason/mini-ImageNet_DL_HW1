import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from model_task_a import ConvNet, DynamicConvNet
from data_loader_rgb_scales import datasets, loaders, mixup_data
import psutil
import pynvml
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# 設置儲存路徑
curve_plot_dir = "/home/wajason99/mini-ImageNet_DL_HW1/curve_plot_A"
os.makedirs(curve_plot_dir, exist_ok=True)

# 設置設備
torch.set_num_threads(12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 GPU 監控
try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except:
    gpu_handle = None

# 計算 FLOPS 和參數量
def get_model_stats(model, input_size):
    from thop import profile
    model.eval()
    input_tensor = torch.randn(1, 3, input_size, input_size).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops / 1e9, params / 1e6  # GFLOPS, M params

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

def train_and_evaluate(model, model_name, size, epochs=30):
    torch.cuda.empty_cache()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=4e-5)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, total_epochs=epochs, eta_min=0)
    
    # 梯度累積設置
    accum_steps = 2
    effective_batch_size = 64
    actual_batch_size = effective_batch_size // accum_steps
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    best_model_state = None

    # tau 退火
    initial_tau = 30.0
    final_tau = 1.0
    tau_steps = 5  # 前 5 個 epoch 退火

    for epoch in tqdm(range(epochs), desc=f"Training {model_name} (Size {size})"):
        # 更新 tau
        if epoch < tau_steps and isinstance(model, DynamicConvNet):
            tau = initial_tau - (initial_tau - final_tau) * (epoch / tau_steps)
            model.set_tau(tau)

        model.train()
        loss_train, acc_train = 0.0, 0.0
        optimizer.zero_grad()
        for i, (images, labels) in enumerate(loaders[size]['train']):
            images, labels = images.to(device), labels.to(device)
            
            # Mixup 數據增強
            if np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=1.0)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0

            outputs = model(images)
            loss = lam * F.cross_entropy(outputs, labels_a) + (1 - lam) * F.cross_entropy(outputs, labels_b)
            loss = loss / accum_steps
            loss.backward()

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_train += loss.item() * accum_steps
            preds = outputs.argmax(dim=1)
            acc = lam * (preds == labels_a).float().mean().item() + (1 - lam) * (preds == labels_b).float().mean().item()
            acc_train += acc

        if (i + 1) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_train /= len(loaders[size]['train'])
        acc_train /= len(loaders[size]['train'])
        train_losses.append(loss_train)
        train_accuracies.append(acc_train)

        model.eval()
        loss_val, acc_val = 0.0, 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(loaders[size]['val'], desc="Validating", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss_val += loss.item()
                preds = outputs.argmax(dim=1)
                acc_val += (preds == labels).float().mean().item()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        loss_val /= len(loaders[size]['val'])
        acc_val /= len(loaders[size]['val'])
        val_losses.append(loss_val)
        val_accuracies.append(acc_val)

        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted', zero_division=0)

        scheduler.step()
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_model_state = model.state_dict()

        print(f"[{model_name}] Epoch {epoch+1}/{epochs} (Size {size}): Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.4f}, "
              f"Val Loss: {loss_val:.4f}, Val Acc: {acc_val:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        print_resource_usage()

    torch.save(best_model_state, os.path.join(curve_plot_dir, f"best_model_{model_name}_size{size}.pt"))

    model.eval()
    test_acc, test_loss = 0.0, 0.0
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in loaders[size]['test']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            test_loss += loss.item()
            preds = outputs.argmax(dim=1)
            test_acc += (preds == labels).float().mean().item()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_loss /= len(loaders[size]['test'])
    test_acc /= len(loaders[size]['test'])
    
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted', zero_division=0)
    
    print(f"[{model_name}] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
          f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, "
          f"Best Val Acc: {best_val_acc:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({model_name}, Size {size})")
    plt.legend()
    plt.savefig(os.path.join(curve_plot_dir, f"loss_curve_{model_name}_size{size}.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label="Train Acc")
    plt.plot(range(1, epochs + 1), val_accuracies, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve ({model_name}, Size {size})")
    plt.legend()
    plt.savefig(os.path.join(curve_plot_dir, f"accuracy_curve_{model_name}_size{size}.png"))
    plt.close()

    return test_acc, best_val_acc, test_precision, test_recall, test_f1

if __name__ == "__main__":
    results = {}
    sizes = [224, 112, 56]

    for size in sizes:
        results[size] = {}
        
        # 訓練 ConvNet
        conv_net = ConvNet(num_classes=50).to(device)
        flops, params = get_model_stats(conv_net, size)
        print(f"\n[ConvNet] Size {size} - FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")
        test_acc, val_acc, test_precision, test_recall, test_f1 = train_and_evaluate(conv_net, f"ConvNet", size, epochs=30)
        results[size]["ConvNet"] = (test_acc, val_acc, test_precision, test_recall, test_f1, flops, params)

        # 訓練 DynamicConvNet
        dynamic_conv_net = DynamicConvNet(num_classes=50, K=4).to(device)
        flops, params = get_model_stats(dynamic_conv_net, size)
        print(f"\n[DynamicConvNet] Size {size} - FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")
        test_acc, val_acc, test_precision, test_recall, test_f1 = train_and_evaluate(dynamic_conv_net, f"DynamicConvNet", size, epochs=30)
        results[size]["DynamicConvNet"] = (test_acc, val_acc, test_precision, test_recall, test_f1, flops, params)

    # 最終性能比較
    for size in sizes:
        print(f"\nFinal Performance Comparison (Size {size}):")
        for model_name in results[size]:
            test_acc, val_acc, test_precision, test_recall, test_f1, flops, params = results[size][model_name]
            print(f"{model_name} Test Acc: {test_acc:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, "
                  f"FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")

    if gpu_handle:
        pynvml.nvmlShutdown()