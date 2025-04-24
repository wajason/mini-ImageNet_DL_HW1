import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from model_task_a import ConvNet, DynamicConvNet, DynamicConvNet_3Layer, DynamicConvNet_2Layer
from data_loader_rgb_scales import datasets, loaders, mixup_data
import psutil
import pynvml
from sklearn.metrics import precision_recall_fscore_support

curve_plot_dir = "/home/wajason99/mini-ImageNet_DL_HW1/curve_plot_A"
os.makedirs(curve_plot_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(15)

try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except:
    gpu_handle = None

def get_model_stats(model, input_size):
    from thop import profile
    model.eval()
    input_tensor = torch.randn(1, 3, input_size, input_size).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops / 1e9, params / 1e6

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

def test_channel_combinations(model_class, test_loaders, size, model_name, state_dict):
    channel_combinations = ['RGB', 'RG', 'GB', 'RB', 'R', 'G', 'B']
    results = {}

    for idx, channels in enumerate(channel_combinations):
        in_channels = len(channels)
        model = model_class(in_channels=in_channels, num_classes=50, K=2).to(device) if model_name != "ConvNet" else model_class(in_channels=in_channels, num_classes=50).to(device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        model.eval()
        test_acc, test_loss = 0.0, 0.0
        test_preds, test_labels = [], []
        test_loader = test_loaders[idx]

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item()
                preds = outputs.argmax(dim=1)
                test_acc += (preds == labels).float().mean().item()
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted', zero_division=0)
        results[channels] = (test_acc, test_precision, test_recall, test_f1)
        print(f"[{model_name}] Size {size} - {channels} Channels: Test Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    return results

def train_and_evaluate(model_class, model_name, size, epochs=5, batch_size=32):
    torch.cuda.empty_cache()
    model = model_class(in_channels=3, num_classes=50, K=2).to(device) if model_name != "ConvNet" else model_class(in_channels=3, num_classes=50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=4e-5)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    best_model_state = None

    initial_tau = 12.0
    final_tau = 1.0
    tau_steps = epochs

    for epoch in tqdm(range(epochs), desc=f"Training {model_name} (Size {size})"):
        if epoch < tau_steps and model_name != "ConvNet":
            tau = initial_tau - (initial_tau - final_tau) * (epoch / tau_steps)
            model.set_tau(tau)

        model.train()
        loss_train, acc_train = 0.0, 0.0
        running_loss, running_acc = 0.0, 0.0
        train_loader = loaders[size]['train']
        total_batches = len(train_loader)

        for batch_count, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=1.0)

            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            acc = lam * (preds == labels_a).float().mean().item() + (1 - lam) * (preds == labels_b).float().mean().item()
            acc_train += acc
            running_acc += acc

            # 每 500 個 batch 或 epoch 結束時顯示進度
            if batch_count % 500 == 0 or batch_count == total_batches:
                avg_loss = running_loss / min(500, batch_count)
                avg_acc = running_acc / min(500, batch_count)
                print(f"[{model_name}] Epoch {epoch+1}, Batch {batch_count}/{total_batches}, "
                      f"Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
                running_loss = 0.0
                running_acc = 0.0

        loss_train /= total_batches
        acc_train /= total_batches
        train_losses.append(loss_train)
        train_accuracies.append(acc_train)

        model.eval()
        loss_val, acc_val = 0.0, 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in loaders[size]['val']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
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

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_model_state = model.state_dict()

        print(f"[{model_name}] Epoch {epoch+1}/{epochs} (Size {size}): Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.4f}, "
              f"Val Loss: {loss_val:.4f}, Val Acc: {acc_val:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        print_resource_usage()

    torch.save(best_model_state, os.path.join(curve_plot_dir, f"best_model_{model_name}_size{size}.pt"))
    channel_results = test_channel_combinations(model_class, loaders[size]['test'], size, model_name, best_model_state)

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

    return acc_val, best_val_acc, val_precision, val_recall, val_f1, channel_results

if __name__ == "__main__":
    for txt_file in ["data/train.txt", "data/val.txt", "data/test.txt"]:
        with open(txt_file, 'r') as f:
            for line in f:
                image_path, _ = line.strip().split()
                assert os.path.exists(os.path.join("data", image_path)), f"Image {image_path} not found!"
    print("All image paths verified successfully.")

    results = {}
    sizes = [224]

    for size in sizes:
        results[size] = {}
        """
        flops, params = get_model_stats(DynamicConvNet(in_channels=3, num_classes=50, K=2).to(device), size)
        print(f"\n[DynamicConvNet] Size {size} - FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")
        test_acc, val_acc, test_precision, test_recall, test_f1, channel_results = train_and_evaluate(DynamicConvNet, "DynamicConvNet", size)
        results[size]["DynamicConvNet"] = (test_acc, val_acc, test_precision, test_recall, test_f1, flops, params, channel_results)

        flops, params = get_model_stats(ConvNet(in_channels=3, num_classes=50).to(device), size)
        print(f"\n[ConvNet] Size {size} - FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")
        test_acc, val_acc, test_precision, test_recall, test_f1, channel_results = train_and_evaluate(ConvNet, "ConvNet", size)
        results[size]["ConvNet"] = (test_acc, val_acc, test_precision, test_recall, test_f1, flops, params, channel_results)
        """
    size = 224
    ablation_results = {}
    flops, params = get_model_stats(DynamicConvNet_3Layer(in_channels=3, num_classes=50, K=2).to(device), size)
    print(f"\n[DynamicConvNet_3Layer] Size {size} - FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")
    test_acc, val_acc, test_precision, test_recall, test_f1, channel_results = train_and_evaluate(DynamicConvNet_3Layer, "DynamicConvNet_3Layer", size)
    ablation_results["DynamicConvNet_3Layer"] = (test_acc, val_acc, test_precision, test_recall, test_f1, flops, params, channel_results)

    flops, params = get_model_stats(DynamicConvNet_2Layer(in_channels=3, num_classes=50, K=2).to(device), size)
    print(f"\n[DynamicConvNet_2Layer] Size {size} - FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")
    test_acc, val_acc, test_precision, test_recall, test_f1, channel_results = train_and_evaluate(DynamicConvNet_2Layer, "DynamicConvNet_2Layer", size)
    ablation_results["DynamicConvNet_2Layer"] = (test_acc, val_acc, test_precision, test_recall, test_f1, flops, params, channel_results)

    for size in sizes:
        print(f"\nFinal Performance Comparison (Size {size}):")
        for model_name in results[size]:
            test_acc, val_acc, test_precision, test_recall, test_f1, flops, params, _ = results[size][model_name]
            print(f"{model_name} Test Acc: {test_acc:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, "
                  f"FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")

    print("\nChannel Combination Comparison (Size 224):")
    for model_name in results[224]:
        print(f"\n{model_name}:")
        channel_results = results[224][model_name][-1]
        for combo_name, (acc, precision, recall, f1) in channel_results.items():
            print(f"{combo_name} - Test Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    print("\nAblation Study (DynamicConvNet Layers, Size 224):")
    for model_name in ["DynamicConvNet", "DynamicConvNet_3Layer", "DynamicConvNet_2Layer"]:
        if model_name == "DynamicConvNet":
            test_acc, val_acc, test_precision, test_recall, test_f1, flops, params, _ = results[224][model_name]
        else:
            test_acc, val_acc, test_precision, test_recall, test_f1, flops, params, _ = ablation_results[model_name]
        print(f"{model_name} Test Acc: {test_acc:.4f}, Val Acc: {val_acc:.4f}, "
              f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, "
              f"FLOPS: {flops:.2f} GFLOPS, Params: {params:.2f} M")

    if gpu_handle:
        pynvml.nvmlShutdown()