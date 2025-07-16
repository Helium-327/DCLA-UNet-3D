# DCLA-UNet: Dynamic Cross Layer Attention U-Net for Medical Image Segmentation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Abstract

DCLA-UNet introduces a novel **Dynamic Cross Layer Attention (DCLA)** mechanism that dynamically selects the most relevant features from different layers of the network. This attention mechanism enhances the feature representation capability of U-Net for medical image segmentation tasks, particularly for brain tumor segmentation in BraTS datasets.


## 🛠️ Overview
![DCLA_UNet_Overview](/assets/DCLA_UNet_Overview.png)
## Results
![对比实验可视化结果](/assets/对比实验可视化结果（v3）.png)
![对比实验可视化结果3d](/assets/对比实验可视化结果（3d）.png)

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone git@github.com:Helium-327/DCLA-UNet-3D.git
cd DCLA-UNet
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Quick Training Example

```bash
# Train DCLA-UNet on BraTS2020 dataset
python src/main.py --model_name "DCLA_UNet_final" \
                   --slb_project "my_experiment" \
                   --datasets "BraTS2020" \
                   --data_root "/path/to/BraTS2020/raw" \
                   --epochs 100 \
                   --batch_size 2 \
                   --lr 3e-4
```

## 📊 Dataset

### Supported Datasets

- **BraTS2019**: Brain Tumor Segmentation Challenge 2019
- **BraTS2020**: Brain Tumor Segmentation Challenge 2020  
- **BraTS2021**: Brain Tumor Segmentation Challenge 2021

### Dataset Structure

Your dataset should be organized as follows:

```
data/
├── BraTS2020/
│   ├── raw/
│   │   ├── BraTS20_Training_001/
│   │   │   ├── BraTS20_Training_001_flair.nii.gz
│   │   │   ├── BraTS20_Training_001_t1.nii.gz
│   │   │   ├── BraTS20_Training_001_t1ce.nii.gz
│   │   │   ├── BraTS20_Training_001_t2.nii.gz
│   │   │   └── BraTS20_Training_001_seg.nii.gz
│   │   └── ...
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
```

### Dataset Preparation

1. **Download BraTS dataset** from the official website
2. **Extract and organize** the data according to the structure above
3. **Generate CSV files** for train/val/test splits:

```bash
python src/main.py --data_split --datasets "BraTS2020" --data_root "/path/to/BraTS2020/raw"
```

## 🏗️ Architecture

### Supported Models

#### SOTA Models
- **UNet3D**: Standard 3D U-Net
- **AttUNet3D**: Attention U-Net 3D
- **UNETR**: Vision Transformer for medical segmentation
- **UNETR_PP**: Enhanced UNETR
- **SegFormer3D**: 3D SegFormer
- **Mamba3d**: State Space Model for 3D segmentation
- **MogaNet**: Multi-order Gated Aggregation Network

#### DCLA-UNet Variants
- **DCLA_UNet_final**: Main DCLA-UNet model
- **BaseLine_S_DCLA_final**: Baseline with DCLA
- **BaseLine_S_DCLA_SLK_final**: DCLA + Selective Large Kernel
- **BaseLine_S_DCLA_MSF_final**: DCLA + Multi-Scale Fusion

### Key Features

- **Dynamic Cross Layer Attention (DCLA)**: Adaptively selects relevant features across different network layers
- **Multi-Scale Fusion (MSF)**: Integrates features at multiple scales
- **Selective Large Kernel (SLK)**: Enhances receptive field with efficient computation
- **Mixed Precision Training**: Accelerated training with automatic mixed precision

## 🔧 Training

### Basic Training

```bash
python src/main.py --model_name "DCLA_UNet_final" \
                   --slb_project "experiment_name" \
                   --datasets "BraTS2020" \
                   --data_root "/path/to/data" \
                   --epochs 100 \
                   --batch_size 2 \
                   --lr 3e-4 \
                   --wd 2e-5
```

### Advanced Training Options

```bash
python src/main.py --model_name "DCLA_UNet_final" \
                   --slb_project "advanced_experiment" \
                   --datasets "BraTS2021" \
                   --data_root "/path/to/data" \
                   --epochs 200 \
                   --batch_size 4 \
                   --lr 3e-4 \
                   --wd 2e-5 \
                   --cosine_eta_min 1e-6 \
                   --cosine_T_max 100 \
                   --early_stop_patience 60 \
                   --slb \
                   --tb
```

### Batch Training Script

Use the provided shell script for training multiple models:

```bash
# Edit run.sh to specify models and parameters
./run.sh
```

### Resume Training

```bash
python src/main.py --resume "/path/to/checkpoint.pth" \
                   --model_name "DCLA_UNet_final" \
                   --slb_project "resumed_experiment"
```

### Training Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--lr` | Learning rate | 3e-4 | 1e-4 to 5e-4 |
| `--wd` | Weight decay | 1e-5 | 1e-5 to 2e-5 |
| `--batch_size` | Batch size | 1 | 2-4 (depends on GPU) |
| `--epochs` | Training epochs | 100 | 100-200 |
| `--early_stop_patience` | Early stopping patience | 60 | 60-100 |
| `--cosine_T_max` | Cosine scheduler T_max | 50 | Half of epochs |

## 📈 Monitoring and Visualization

### Training Monitoring

- **SwanLab**: Use `--slb` flag for experiment tracking
- **TensorBoard**: Use `--tb` flag for TensorBoard logging

### Results Visualization

The project includes comprehensive visualization tools:

```bash
# Launch Gradio interface for result visualization
python visual/gradio_visual_v2.py
```

### Available Visualizations

1. **Multi-modal data visualization** (`visual/1_多模态数据样本可视化.ipynb`)
2. **Attention map visualization** (`visual/2_注意力图可视化.ipynb`)
3. **Multi-model comparison** (`visual/3_多模型结果可视化.ipynb`)
4. **Single model multi-modal analysis** (`visual/4_单模型多模态可视化.ipynb`)

## 🔬 Evaluation Metrics

The framework supports comprehensive evaluation metrics:

- **Dice Score**: Overlap-based similarity measure
- **Hausdorff Distance**: Boundary-based distance measure
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **Volume Similarity**: Volume-based comparison

## 🛠️ How to Add a New Model?

1. **Create model file** in `src/nnArchitecture/nets/`:

```python
# src/nnArchitecture/nets/your_model.py
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        # Your model implementation
        
    def forward(self, x):
        # Forward pass
        return x
```

2. **Register model** in `src/model_registry.py`:

```python
from nnArchitecture.nets.your_model import YourModel

model_register = {
    "Your Models": {
        "YourModel": YourModel(**BASE_ARGS),
    }
}
```

3. **Update imports** in `src/nnArchitecture/nets/__init__.py`:

```python
from .your_model import YourModel
```

## 📁 How to Add a New Dataset?

1. **Create dataset directory**: `src/datasets/YourDataset/`

2. **Implement dataset class**:

```python
# src/datasets/YourDataset/your_dataset.py
from torch.utils.data import Dataset

class YourDataset(Dataset):
    def __init__(self, data_file, transform=None):
        # Dataset initialization
        
    def __getitem__(self, idx):
        # Return data sample
        
    def __len__(self):
        # Return dataset length
```

3. **Update main.py** to support your dataset:

```python
if args.datasets == 'YourDataset':
    from datasets.YourDataset import YourDataset as Dataset
```

## 📋 Project Structure

```
DCLA-UNet/
├── src/
│   ├── datasets/           # Dataset implementations
│   ├── nnArchitecture/     # Model architectures
│   ├── utils/              # Utility functions
│   ├── main.py            # Main training script
│   ├── train_swanlab.py   # Training with SwanLab
│   ├── metrics.py         # Evaluation metrics
│   └── lossFunc.py        # Loss functions
├── visual/                # Visualization tools
├── requirements.txt       # Dependencies
├── run.sh                # Training script
└── README.md             # This file
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size: `--batch_size 1`
   - Use gradient checkpointing
   - Reduce input size

2. **Dataset loading errors**:
   - Check file paths in CSV files
   - Verify data directory structure
   - Ensure all required modalities are present

3. **Training instability**:
   - Reduce learning rate: `--lr 1e-4`
   - Increase weight decay: `--wd 2e-5`
   - Use gradient clipping

### Performance Optimization

- Use `--num_workers 8` for faster data loading
- Enable mixed precision training (automatically enabled)
- Use `persistent_workers=True` for DataLoader

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{dcla_unet_2024,
  title={DCLA-UNet: Dynamic Cross Layer Attention U-Net for Medical Image Segmentation},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions and support, please contact:
- Email: your.email@example.com
- GitHub Issues: [Create an issue](https://github.com/your-username/DCLA-UNet/issues)

---

**Note**: This project is under active development. Please check the latest updates and documentation regularly.

