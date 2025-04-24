# Mini-ImageNet Deep Learning Homework 1

This repository contains the implementation for Homework 1 of the Deep Learning course, focusing on image classification using the mini-ImageNet dataset. The project includes two tasks:

- **Task A**: Implementation of `ConvNet` and `DynamicConvNet` models, training with Mixup augmentation, and evaluation on different channel combinations (RGB, RG, GB, RB, R, G, B).
- **Task B**: [Placeholder - Implementation of additional models or tasks, e.g., advanced architectures or techniques; replace with specific Task B description].

The code is implemented in Python using PyTorch and evaluated on an NVIDIA RTX 3090 GPU.

## Repository Structure

```
mini-ImageNet_DL_HW1/
├── src/
│   ├── model_task_a.py         # Model definitions for Task A (ConvNet, DynamicConvNet, etc.)
│   ├── train_task_a.py         # Training and evaluation script for Task A
│   ├── data_loader_rgb_scales.py  # Data loader for mini-ImageNet with channel combinations
│   ├── model_task_b.py         # Model definitions for Task B (placeholder)
│   ├── train_task_b.py         # Training and evaluation script for Task B (placeholder)
├── data/
│   ├── train.txt               # Training image paths and labels
│   ├── val.txt                 # Validation image paths and labels
│   ├── test.txt                # Test image paths and labels
│   ├── images/                 # Mini-ImageNet image files
├── curve_plot_A/               # Output directory for Task A training curves
├── curve_plot_B/               # Output directory for Task B training curves (placeholder)
├── README.md                   # This file
```

## Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (e.g., RTX 3090) recommended for faster training.
- **CPU**: Multi-core CPU for data loading.
- **Memory**: At least 16GB RAM and 12GB GPU memory.

### Software
- **Python**: 3.11
- **Conda**: For environment management.
- **Dependencies**:
  - `torch==2.0.0`
  - `torchvision==0.15.0`
  - `thop==0.1.1`
  - `matplotlib==3.7.0`
  - `tqdm==4.65.0`
  - `psutil==5.9.0`
  - `pynvml==11.5.0`
  - `scikit-learn==1.2.0`

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/mini-ImageNet_DL_HW1.git
   cd mini-ImageNet_DL_HW1
   ```

2. **Create and Activate Conda Environment**
   ```bash
   conda create -n dl_hw1 python=3.11
   conda activate dl_hw1
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirement.py
   ```

4. **Prepare Mini-ImageNet Dataset**
   - Ensure the structure is:
     ```
     data/
     ├── train.txt
     ├── val.txt
     ├── test.txt
     ├── images/
     ```
   - Each `.txt` file should contain image paths (relative to `data/`) and labels, e.g.:
     ```
     images/n01532829/n0153282900000001.jpg 0
     ```
   - Verify image paths:
     ```bash
     python -c "from src.train_task_a import *; print('Image paths verified')"
     ```

## Running Task A

Task A trains and evaluates `ConvNet`, `DynamicConvNet`, `DynamicConvNet_3Layer`, and `DynamicConvNet_2Layer` on mini-ImageNet with image size 224x224, using Mixup augmentation. It evaluates performance on different channel combinations (RGB, RG, GB, RB, R, G, B) and saves training curves.

### Instructions
1. **Navigate to the Source Directory**
   ```bash
   cd src
   ```

2. **Run the Training Script**
   ```bash
   python train_task_a.py
   ```
   - **Parameters**:
     - `epochs=10`
     - `batch_size=48`
     - `lr=0.001`
     - `mixup_alpha=0.2`
   - **Outputs**:
     - Training logs with loss, accuracy, and resource usage.
     - Best model weights saved to `curve_plot_A/best_model_[model_name]_size224.pt`.
     - Loss and accuracy curves saved to `curve_plot_A/loss_curve_[model_name]_size224.png` and `curve_plot_A/accuracy_curve_[model_name]_size224.png`.
     - Performance metrics (Test Acc, Precision, Recall, F1) for each model and channel combination.

3. **Expected Results**
   - **ConvNet**: Test Acc ~0.1271, FLOPS ~0.40 GFLOPS, Params ~0.40 M
   - **DynamicConvNet**: Test Acc ~0.12-0.14, FLOPS ~0.45 GFLOPS, Params ~0.80 M
   - **DynamicConvNet_3Layer**: Test Acc ~0.10-0.12
   - **DynamicConvNet_2Layer**: Test Acc ~0.08-0.10
   - Channel combination results (RGB best, followed by RG/GB/RB, then R/G/B).

4. **Training Time**
   - ~1-2 hours on RTX 3090 for all models.

## Running Task B

Task B involves [Placeholder - e.g., training advanced models or different techniques on mini-ImageNet; replace with specific Task B description]. The implementation is in `model_task_b.py` and `train_task_b.py`.

### Instructions
1. **Navigate to the Source Directory**
   ```bash
   cd src
   ```

2. **Run the Training Script**
   ```bash
   python train_task_b.py
   ```
   - **Parameters** (adjust based on Task B specifics):
     - `epochs=10`
     - `batch_size=48`
     - `lr=0.001`
   - **Outputs**:
     - Training logs with performance metrics.
     - Model weights saved to `curve_plot_B/best_model_[model_name]_size224.pt`.
     - Plots saved to `curve_plot_B/`.

3. **Expected Results**
   - [Placeholder - Replace with expected metrics, e.g., Test Acc, FLOPS, Params for Task B models].

4. **Training Time**
   - ~1-2 hours on RTX 3090 (adjust based on Task B complexity).

## Reproducing Experiments

To reproduce the experiments exactly as reported:

1. **Ensure Environment Matches**
   - Use Python 3.11, PyTorch 2.0.0, and listed dependencies.
   - Run on an NVIDIA GPU with CUDA 11.7 or later.

2. **Verify Dataset**
   - Confirm mini-ImageNet images and `.txt` files are correctly placed in `data/`.
   - Check data loader (`data_loader_rgb_scales.py`) applies normalization:
     ```python
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ```

3. **Run Scripts Sequentially**
   - Task A: `python src/train_task_a.py`
   - Task B: `python src/train_task_b.py`
   - Check output logs and plots in `curve_plot_A/` and `curve_plot_B/`.

4. **Inspect Results**
   - Compare Test Acc, Precision, Recall, F1, FLOPS, and Params with reported values.
   - Verify training curves for convergence (loss decreasing, accuracy increasing).

## Troubleshooting

- **Error: Image paths not found**
  - Ensure `data/images/` contains mini-ImageNet images and `.txt` files have correct paths.
  - Run path verification:
    ```bash
    python src/train_task_a.py
    ```

- **Error: Out of GPU memory**
  - Reduce `batch_size` to 32 or 16 in `train_task_a.py` or `train_task_b.py`:
    ```python
    def train_and_evaluate(model_class, model_name, size, epochs=10, batch_size=32):
    ```
  - Clear GPU memory:
    ```bash
    nvidia-smi
    kill -9 <PID>
    ```

- **Low Test Accuracy**
  - Increase `epochs` to 15 or 20.
  - Lower `mixup_alpha` to 0.1 in `train_task_a.py`:
    ```python
    images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.1)
    ```
  - Verify data normalization in `data_loader_rgb_scales.py`.

- **FLOPS Calculation Fails**
  - Update `thop`:
    ```bash
    pip install thop --upgrade
    ```
  - Manually compute parameters:
    ```python
    def manual_params(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params / 1e6
    ```

## Contact

For issues or questions, please contact [your-email@example.com] or open an issue on GitHub.
