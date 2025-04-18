import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_task_b import SimpleCNN
from data_loader import train_loader, val_loader, test_loader
from torchvision.models import resnet34
import psutil
import os
import pynvml

# 限制 PyTorch 線程數和 GPU 設置
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 GPU 監控
try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except:
    gpu_handle = None

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

def train_and_evaluate(model, model_name="SimpleCNN"):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in tqdm(range(epochs)):
        model.train()
        loss_train, acc_train = 0.0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            preds = outputs.argmax(dim=1)
            acc_train += (preds == labels).sum().item() / len(labels)

        loss_train /= len(train_loader)
        acc_train /= len(train_loader)
        train_losses.append(loss_train)
        train_accuracies.append(acc_train)

        model.eval()
        loss_val, acc_val = 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss_val += loss.item()
                preds = outputs.argmax(dim=1)
                acc_val += (preds == labels).sum().item() / len(labels)

        loss_val /= len(val_loader)
        acc_val /= len(val_loader)
        val_losses.append(loss_val)
        val_accuracies.append(acc_val)

        print(f"[{model_name}] Epoch {epoch+1}: Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.4f}, Val Loss: {loss_val:.4f}, Val Acc: {acc_val:.4f}")
        print_resource_usage()

    model.eval()
    test_acc, test_loss = 0.0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            test_loss += loss.item()
            preds = outputs.argmax(dim=1)
            test_acc += (preds == labels).sum().item() / len(labels)

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f"[{model_name}] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({model_name})")
    plt.legend()
    plt.savefig(f"task_b_loss_curve_{model_name}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label="Train Acc")
    plt.plot(range(1, epochs + 1), val_accuracies, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve ({model_name})")
    plt.legend()
    plt.savefig(f"task_b_accuracy_curve_{model_name}.png")

    return test_acc

if __name__ == "__main__":
    simple_cnn = SimpleCNN(num_classes=1000).to(device)
    simple_cnn_acc = train_and_evaluate(simple_cnn, "SimpleCNN")

    resnet = resnet34(pretrained=False, num_classes=1000).to(device)
    resnet_acc = train_and_evaluate(resnet, "ResNet34")

    print(f"\nPerformance Comparison:")
    print(f"SimpleCNN Test Acc: {simple_cnn_acc:.4f}")
    print(f"ResNet34 Test Acc: {resnet_acc:.4f}")
    print(f"SimpleCNN / ResNet34: {(simple_cnn_acc / resnet_acc) * 100:.2f}%")

    if gpu_handle:
        pynvml.nvmlShutdown()