import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_task_a import TaskAModel
from model_naive import NaiveModel
from data_loader import train_loader, val_loader, test_loader
import psutil
import os
import pynvml
from ptflops import get_model_complexity_info
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# 修復 torch._dynamo 錯誤
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # 抑制 torch._dynamo 錯誤，回退到 eager 模式

torch.manual_seed(66)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(66)

torch.set_num_threads(12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except:
    gpu_handle = None

curve_plot_dir = "/home/wajason99/mini-ImageNet_DL_HW1/curve_plot"
os.makedirs(curve_plot_dir, exist_ok=True)

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

def compute_complexity(model, in_channels, channel_name):
    model.eval()
    input_shape = (1, in_channels, 224, 224)
    try:
        input_tensor = torch.randn(input_shape, requires_grad=True).to(device)
        flops, params = get_model_complexity_info(
            model,
            (in_channels, 224, 224),
            as_strings=True,
            print_per_layer_stat=False,
            input_constructor=lambda _: input_tensor
        )
        print(f"[{channel_name}] Complexity: FLOPS: {flops}, Parameters: {params}")
    except Exception as e:
        print(f"FLOPS calculation failed: {e}")
        flops, params = "N/A", "N/A"
    return flops, params

def plot_confusion_matrix(y_true, y_pred, channel_name, model_type):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix ({channel_name} - {model_type})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(curve_plot_dir, f"confusion_matrix_{channel_name}_{model_type}.png"))
    plt.close()

def train_and_evaluate(in_channels=3, channel_name="RGB", use_dynamic=True, num_layers=5, epochs=35, num_experiments=1):
    test_accuracies, test_losses = [], []
    precisions, recalls, f1_scores = [], [], []
    all_flops, all_params = [], []

    for exp in range(num_experiments):
        if use_dynamic:
            model = TaskAModel(in_channels=in_channels, num_classes=50, num_layers=num_layers).to(device)
            model_type = f"Dynamic_Layers{num_layers}"
        else:
            model = NaiveModel(in_channels=in_channels, num_classes=50, num_layers=num_layers).to(device)
            model_type = f"Naive_Layers{num_layers}"

        flops, params = compute_complexity(model, in_channels, f"{channel_name} ({model_type})")
        all_flops.append(flops)
        all_params.append(params)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5, min_lr=0.00001, verbose=True)
        scaler = GradScaler()

        patience = 15  # 早停的 patience
        best_val_acc = 0.0
        counter = 0
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        best_model_path = os.path.join(curve_plot_dir, f"best_model_{channel_name}_{model_type}.pt")

        for epoch in tqdm(range(epochs), desc=f"Exp {exp+1} Training {channel_name} ({model_type})"):
            model.train()
            loss_train, acc_train, total_train = 0.0, 0.0, 0
            for images, labels in train_loader:
                images = images[:, :in_channels, :, :]
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with autocast():
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_train += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                acc_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            loss_train /= total_train
            acc_train /= total_train
            train_losses.append(loss_train)
            train_accuracies.append(acc_train)

            model.eval()
            loss_val, acc_val, total_val = 0.0, 0.0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images[:, :in_channels, :, :]
                    images, labels = images.to(device), labels.to(device)
                    with autocast():
                        outputs = model(images)
                        loss = F.cross_entropy(outputs, labels)
                    loss_val += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    acc_val += (preds == labels).sum().item()
                    total_val += labels.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            loss_val /= total_val
            acc_val /= total_val
            val_losses.append(loss_val)
            val_accuracies.append(acc_val)

            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

            print(f"[{channel_name}] ({model_type}) Exp {exp+1} Epoch {epoch+1}: "
                  f"Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.4f}, "
                  f"Val Loss: {loss_val:.4f}, Val Acc: {acc_val:.4f}, "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, "
                  f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print_resource_usage()

            # 使用 ReduceLROnPlateau，根據 Val Acc 調整學習率
            scheduler.step(acc_val)

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model at {best_model_path} with Val Acc: {best_val_acc:.4f}")
            else:
                counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_acc, test_loss, total_test = 0.0, 0.0, 0
        test_preds, test_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images[:, :in_channels, :, :]
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                test_acc += (preds == labels).sum().item()
                total_test += labels.size(0)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_loss /= total_test
        test_acc /= total_test
        test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
        test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
        test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        precisions.append(test_precision)
        recalls.append(test_recall)
        f1_scores.append(test_f1)

        if exp == num_experiments - 1:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
            plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Loss Curve ({channel_name} - {model_type})")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(curve_plot_dir, f"loss_curve_{channel_name}_{model_type}.png"))
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Acc")
            plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Val Acc")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy Curve ({channel_name} - {model_type})")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(curve_plot_dir, f"accuracy_curve_{channel_name}_{model_type}.png"))
            plt.close()

            plot_confusion_matrix(test_labels, test_preds, channel_name, model_type)

    mean_test_acc = np.mean(test_accuracies)
    std_test_acc = np.std(test_accuracies)
    mean_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)
    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    return (mean_test_acc, std_test_acc, mean_test_loss, std_test_loss,
            mean_precision, std_precision, mean_recall, std_recall, mean_f1, std_f1,
            all_flops[0], all_params[0])

if __name__ == "__main__":
    channel_configs = [
        (3, "RGB"),
        (2, "RG"),
        (2, "RB"),
        (2, "GB"),
        (1, "R"),
        (1, "G"),
        (1, "B")
    ]

    layer_configs = [3, 5, 7]

    results_dynamic = {}
    results_naive = {}

    for in_channels, channel_name in channel_configs:
        print(f"\n=== Training with {channel_name} channels ===")
        
        for use_dynamic in [True, False]:
            model_type = "Dynamic" if use_dynamic else "Naive"
            print(f"\nTraining {model_type} Model...")

            for num_layers in layer_configs:
                print(f"\nAblation Study: {channel_name} ({model_type}) - Layers: {num_layers}")
                metrics = train_and_evaluate(
                    in_channels=in_channels,
                    channel_name=channel_name,
                    use_dynamic=use_dynamic,
                    num_layers=num_layers,
                    epochs=35,
                    num_experiments=1
                )
                (mean_test_acc, std_test_acc, mean_test_loss, std_test_loss,
                 mean_precision, std_precision, mean_recall, std_recall,
                 mean_f1, std_f1, flops, params) = metrics

                key = (channel_name, model_type, num_layers)
                if use_dynamic:
                    results_dynamic[key] = metrics
                else:
                    results_naive[key] = metrics

                print(f"[{channel_name}] ({model_type}) Results: "
                      f"Test Acc: {mean_test_acc:.4f} ± {std_test_acc:.4f}, "
                      f"Test Loss: {mean_test_loss:.4f} ± {std_test_loss:.4f}, "
                      f"Precision: {mean_precision:.4f} ± {std_precision:.4f}, "
                      f"Recall: {mean_recall:.4f} ± {std_recall:.4f}, "
                      f"F1: {mean_f1:.4f} ± {std_f1:.4f}, "
                      f"FLOPS: {flops}, Parameters: {params}")

    print("\n=== Summary of Results ===")
    print("\nDynamic Model:")
    for key, metrics in results_dynamic.items():
        channel_name, model_type, num_layers = key
        mean_test_acc, std_test_acc, _, _, mean_precision, std_precision, mean_recall, std_recall, mean_f1, std_f1, flops, params = metrics
        print(f"{channel_name} (Layers: {num_layers}): "
              f"Test Acc: {mean_test_acc:.4f} ± {std_test_acc:.4f}, "
              f"Precision: {mean_precision:.4f} ± {std_precision:.4f}, "
              f"Recall: {mean_recall:.4f} ± {std_recall:.4f}, "
              f"F1: {mean_f1:.4f} ± {std_f1:.4f}, "
              f"FLOPS: {flops}, Parameters: {params}")

    print("\nNaive Model:")
    for key, metrics in results_naive.items():
        channel_name, model_type, num_layers = key
        mean_test_acc, std_test_acc, _, _, mean_precision, std_precision, mean_recall, std_recall, mean_f1, std_f1, flops, params = metrics
        print(f"{channel_name} (Layers: {num_layers}): "
              f"Test Acc: {mean_test_acc:.4f} ± {std_test_acc:.4f}, "
              f"Precision: {mean_precision:.4f} ± {std_precision:.4f}, "
              f"Recall: {mean_recall:.4f} ± {std_recall:.4f}, "
              f"F1: {mean_f1:.4f} ± {std_f1:.4f}, "
              f"FLOPS: {flops}, Parameters: {params}")

    if gpu_handle:
        pynvml.nvmlShutdown()