# DCLA-UNet: 用于医学图像分割的动态跨层注意力U-Net

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 摘要

DCLA-UNet 使用了深度轴向分离卷积构建了轻量化的编解码器结构，并引入了一种新颖的**动态跨层注意力（DCLA）**机制，能够动态选择网络不同层中最相关的特征。这种注意力机制增强了U-Net在医学图像分割任务中的特征表示能力，特别是在BraTS数据集的脑肿瘤分割任务中表现出色。

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 支持CUDA的GPU（推荐）
- 16GB+ 内存

### 安装

1. **克隆仓库**
```bash
git clone git@github.com:Helium-327/DCLA-UNet-3D.git
cd DCLA-UNet
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

### 快速训练示例

```bash
# 在BraTS2020数据集上训练DCLA-UNet
python src/main.py --model_name "DCLA_UNet_final" \
                   --slb_project "my_experiment" \
                   --datasets "BraTS2020" \
                   --data_root "/path/to/BraTS2020/raw" \
                   --epochs 100 \
                   --batch_size 2 \
                   --lr 3e-4
```

## 📊 数据集

### 支持的数据集

- **BraTS2019**: 2019年脑肿瘤分割挑战赛数据集
- **BraTS2020**: 2020年脑肿瘤分割挑战赛数据集
- **BraTS2021**: 2021年脑肿瘤分割挑战赛数据集

### 数据集结构

您的数据集应按以下方式组织：

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

### 数据集准备

1. **下载BraTS数据集** 从官方网站下载
2. **提取并组织** 按照上述结构组织数据
3. **生成CSV文件** 用于训练/验证/测试集划分：

```bash
python src/main.py --data_split --datasets "BraTS2020" --data_root "/path/to/BraTS2020/raw"
```

## 🏗️ 架构

### 支持的模型

#### SOTA模型
- **UNet3D**: 标准3D U-Net
- **AttUNet3D**: 注意力U-Net 3D
- **UNETR**: 用于医学分割的视觉Transformer
- **UNETR_PP**: 增强版UNETR
- **SegFormer3D**: 3D SegFormer
- **Mamba3d**: 用于3D分割的状态空间模型
- **MogaNet**: 多阶门控聚合网络

#### DCLA-UNet变体
- **DCLA_UNet_final**: 主要的DCLA-UNet模型
- **BaseLine_S_DCLA_final**: 带有DCLA的基线模型
- **BaseLine_S_DCLA_SLK_final**: DCLA + 选择性大核
- **BaseLine_S_DCLA_MSF_final**: DCLA + 多尺度融合

### 关键特性

- **动态跨层注意力（DCLA）**: 自适应地选择不同网络层的相关特征
- **多尺度融合（MSF）**: 在多个尺度上整合特征
- **选择性大核（SLK）**: 通过高效计算增强感受野
- **混合精度训练**: 使用自动混合精度加速训练

## 🔧 训练

### 基础训练

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

### 高级训练选项

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

### 批量训练脚本

使用提供的shell脚本训练多个模型：

```bash
# 编辑run.sh以指定模型和参数
./run.sh
```

### 恢复训练

```bash
python src/main.py --resume "/path/to/checkpoint.pth" \
                   --model_name "DCLA_UNet_final" \
                   --slb_project "resumed_experiment"
```

### 训练参数

| 参数 | 描述 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--lr` | 学习率 | 3e-4 | 1e-4 到 5e-4 |
| `--wd` | 权重衰减 | 1e-5 | 1e-5 到 2e-5 |
| `--batch_size` | 批次大小 | 1 | 2-4（取决于GPU） |
| `--epochs` | 训练轮数 | 100 | 100-200 |
| `--early_stop_patience` | 早停耐心值 | 60 | 60-100 |
| `--cosine_T_max` | 余弦调度器T_max | 50 | 轮数的一半 |

## 📈 监控和可视化

### 训练监控

- **SwanLab**: 使用 `--slb` 标志进行实验跟踪
- **TensorBoard**: 使用 `--tb` 标志进行TensorBoard日志记录

### 结果可视化

项目包含全面的可视化工具：

```bash
# 启动Gradio界面进行结果可视化
python visual/gradio_visual_v2.py
```

### 可用的可视化

1. **多模态数据可视化** (`visual/1_多模态数据样本可视化.ipynb`)
2. **注意力图可视化** (`visual/2_注意力图可视化.ipynb`)
3. **多模型比较** (`visual/3_多模型结果可视化.ipynb`)
4. **单模型多模态分析** (`visual/4_单模型多模态可视化.ipynb`)

## 🔬 评估指标

框架支持全面的评估指标：

- **Dice分数**: 基于重叠的相似性度量
- **Hausdorff距离**: 基于边界的距离度量
- **敏感性**: 真阳性率
- **特异性**: 真阴性率
- **体积相似性**: 基于体积的比较

## 🛠️ 如何添加新模型？

1. **在 `src/nnArchitecture/nets/` 中创建模型文件**：

```python
# src/nnArchitecture/nets/your_model.py
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        # 您的模型实现
        
    def forward(self, x):
        # 前向传播
        return x
```

2. **在 `src/model_registry.py` 中注册模型**：

```python
from nnArchitecture.nets.your_model import YourModel

model_register = {
    "Your Models": {
        "YourModel": YourModel(**BASE_ARGS),
    }
}
```

3. **在 `src/nnArchitecture/nets/__init__.py` 中更新导入**：

```python
from .your_model import YourModel
```

## 📁 如何添加新数据集？

1. **创建数据集目录**: `src/datasets/YourDataset/`

2. **实现数据集类**：

```python
# src/datasets/YourDataset/your_dataset.py
from torch.utils.data import Dataset

class YourDataset(Dataset):
    def __init__(self, data_file, transform=None):
        # 数据集初始化
        
    def __getitem__(self, idx):
        # 返回数据样本
        
    def __len__(self):
        # 返回数据集长度
```

3. **更新main.py以支持您的数据集**：

```python
if args.datasets == 'YourDataset':
    from datasets.YourDataset import YourDataset as Dataset
```

## 📋 项目结构

```
DCLA-UNet/
├── src/
│   ├── datasets/           # 数据集实现
│   ├── nnArchitecture/     # 模型架构
│   ├── utils/              # 工具函数
│   ├── main.py            # 主训练脚本
│   ├── train_swanlab.py   # SwanLab训练
│   ├── metrics.py         # 评估指标
│   └── lossFunc.py        # 损失函数
├── visual/                # 可视化工具
├── requirements.txt       # 依赖项
├── run.sh                # 训练脚本
└── README.md             # 本文件
```

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**：
   - 减少批次大小：`--batch_size 1`
   - 使用梯度检查点
   - 减少输入尺寸

2. **数据集加载错误**：
   - 检查CSV文件中的文件路径
   - 验证数据目录结构
   - 确保所有必需的模态都存在

3. **训练不稳定**：
   - 降低学习率：`--lr 1e-4`
   - 增加权重衰减：`--wd 2e-5`
   - 使用梯度裁剪

### 性能优化

- 使用 `--num_workers 8` 加快数据加载
- 启用混合精度训练（自动启用）
- 为DataLoader使用 `persistent_workers=True`

## 📄 引用

如果您在研究中使用此代码，请引用：

```bibtex
@article{dcla_unet_2024,
  title={DCLA-UNet: Dynamic Cross Layer Attention U-Net for Medical Image Segmentation},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 📜 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎贡献！请随时提交Pull Request。

## 📞 联系方式

如有问题和支持，请联系：
- 邮箱：your.email@example.com
- GitHub Issues：[创建issue](https://github.com/your-username/DCLA-UNet/issues)

---

**注意**：本项目正在积极开发中。请定期查看最新更新和文档。