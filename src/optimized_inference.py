# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/24 15:49:32
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 优化之后的推理代码
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''

import os
import time
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, Any, Optional

from datasets.BraTS21 import BraTS21_3D
from datasets.transforms import CenterCrop3D, Compose, FrontGroundNormalize, RandomCrop3D, ToTensor
from metrics import *
from utils.ckpt_tools import load_checkpoint
from train_init import load_model

Tensor = torch.Tensor
Model = torch.nn.Module
DataLoader = torch.utils.data.DataLoader
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AdamW = torch.optim.AdamW
GradScaler = torch.amp.GradScaler
autocast = torch.amp.autocast('cuda')



def load_data(test_csv, local_train=False, test_length=10, batch_size=1, num_workers=4):
    """加载数据集"""
    TransMethods_test = Compose([
        RandomCrop3D(size=(155, 240, 240)),
        FrontGroundNormalize(),
        ToTensor(),
    ])

    test_dataset = BraTS21_3D(
        data_file=test_csv,
        transform=TransMethods_test,
        local_train=local_train,
        length=test_length,
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True  # 减少 worker 初始化时间
    )
    
    print(f"已加载测试数据: {len(test_loader)}")
    return test_loader


def inference(
    test_df: pd.DataFrame,
    test_loader: torch.utils.data.DataLoader,
    output_root: str,
    model: torch.nn.Module,
    metricer: EvaluationMetrics,
    scaler: torch.cuda.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    ckpt_path: str,
    affine: Optional[np.ndarray] = None,
    window_size: list[int, int, int] | int = 128,
    stride_ratio: float = 0.5,
    save_flag: bool = True,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Dict[str, float]:
    """
    医学影像推理主函数
    
    参数说明：
    test_df: 包含病例元数据的DataFrame
    test_loader: 测试数据加载器
    output_root: 输出根目录
    model: 训练好的模型
    metricer: 指标计算器
    ...（其他参数说明）
    """
    # 初始化配置
    affine = affine or default_affine()
    start_time = time.time()
    
    # 模型加载优化
    model, optimizer, scaler, _, _, _ = load_checkpoint(model, optimizer, scaler, ckpt_path, load_weights_only=False)
    model.to(device)
    
    # 创建输出目录
    output_path, new_ckpt_path = init_output_dir(output_root, model, ckpt_path)
    
    # 异步执行器
    executor = ThreadPoolExecutor(max_workers=4)
    
    # 指标容器
    metrics_accumulator = torch.zeros((5, 4), dtype=torch.float32, device=device)
    
    for i, data in enumerate(tqdm(test_loader, desc="推理进度")):
        vimage, vmask = data[0], data[1]
        
        # 半精度推理优化[10](@ref)
        with torch.no_grad():
            pred_vimage = slide_window_pred(vimage.half(), model, window_size, stride_ratio)
        
        case_id = test_df.iloc[i]['patient_idx']
        
        # 异步保存结果
        if save_flag:
            future = executor.submit(
                async_save_results,
                test_df, pred_vimage, vmask, output_path, affine, case_id
            )
            future.add_done_callback(lambda x: print(f"案例 {case_id} 保存完成") if x.exception() is None else None)
        
        # 指标计算
        batch_metrics = metricer.update(y_pred=pred_vimage, y_mask=vmask, hd95=True)
        metrics_accumulator += batch_metrics
        
        # 内存优化[9](@ref)
        del pred_vimage, vmask
        torch.cuda.empty_cache()
    
    # 生成报告
    final_metrics = process_metrics(metrics_accumulator / len(test_loader))
    generate_report(
        model_name=model.__class__.__name__,
        ckpt_path=new_ckpt_path,
        inference_time=time.time() - start_time,
        metrics=final_metrics,
        output_path=output_path
    )
    
    # return final_metrics

def slide_window_pred(inputs, model, roi_size=128, overlap=0, mode="constant"):  
    #TODO: 需要添加多batch支持
    """
    BraTS专用高效滑窗推理函数
    参数：
        inputs: 输入张量 (b, 4, 155, 240, 240)
        model: 训练好的分割模型
        roi_size: 窗口大小，默认128x128x128
        sw_batch_size: 滑动窗口批大小
        overlap: 窗口重叠率
        mode: 融合模式("gaussian"/"constant")
    """
    print(f"滑窗推理: {inputs.shape} -> {roi_size}x{roi_size}x{roi_size}")
    print(f"重叠率: {overlap}")
    print(f"融合模式: {mode}")
    # 设备配置
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    
    # 计算滑动步长
    strides = [int(roi_size * (1 - overlap))] * 3
    strides[0] = roi_size  # 深度方向不重叠
    
    # 生成滑动窗口坐标
    dims = inputs.shape[2:]

    num_blocks = [int(np.ceil(d / s)) for d, s in zip(dims, strides)]
    
    # 初始化输出概率图和计数图
    output_map = torch.zeros((inputs.shape[0], 4, *dims), device=device)
    count_map = torch.zeros((1, 1, *dims), device=device)
    
    # 生成高斯权重窗口
    if mode == "gaussian":
        sigma = 0.125 * roi_size
        coords = torch.arange(roi_size, device=device).float()
        grid = torch.stack(torch.meshgrid(coords, coords, coords), dim=-1)
        center = roi_size // 2
        weights = torch.exp(-torch.sum((grid - center)**2, dim=-1) / (2 * sigma**2))
        weights = weights / weights.max()
    else:
        weights = torch.ones((roi_size, roi_size, roi_size), device=device)
    
    # 滑窗推理
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for d in tqdm(range(num_blocks[0])):
            for h in range(num_blocks[1]):
                for w in range(num_blocks[2]):
                    # 计算当前窗口坐标
                    d_start = min(d * strides[0], dims[0] - roi_size)
                    h_start = min(h * strides[1], dims[1] - roi_size)
                    w_start = min(w * strides[2], dims[2] - roi_size)
                    
                    # 提取窗口数据
                    window = inputs[
                        :, :,
                        d_start:d_start+roi_size,
                        h_start:h_start+roi_size,
                        w_start:w_start+roi_size
                    ]
                    
                    # 模型推理
                    pred = model(window)
                    
                    # 加权融合
                    output_map[
                        :, :,
                        d_start:d_start+roi_size,
                        h_start:h_start+roi_size,
                        w_start:w_start+roi_size
                    ] += pred * weights
                    
                    count_map[
                        :, :,
                        d_start:d_start+roi_size,
                        h_start:h_start+roi_size,
                        w_start:w_start+roi_size
                    ] += weights
                    
                    # 显存清理
                    del window, pred
                    if w % 4 == 0:
                        torch.cuda.empty_cache()
    
    # 归一化输出
    output_map /= count_map
    return output_map.cpu()


def init_output_dir(output_root: str, model: Model, ckpt_path: str) -> Tuple[str, str]:
    """初始化输出目录结构"""
    timestamp = f"{pd.Timestamp.now():%Y%m%d%H%M%S}"
    output_path = os.path.join(output_root, f"{model.__class__.__name__}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # 复制检查点文件
    new_ckpt_path = shutil.copy(ckpt_path, os.path.join(output_path, os.path.basename(ckpt_path)))
    return output_path, new_ckpt_path

def async_save_results(
    test_df: pd.DataFrame,
    pred_vimage: torch.Tensor,
    vmask: torch.Tensor,
    output_path: str,
    affine: np.ndarray,
    case_id: str
) -> None:
    """异步保存推理结果"""
    try:
        # 生成预测结果
        test_output_argmax = torch.argmax(pred_vimage, dim=1).to(torch.int64)
        
        # 保存NIfTI文件
        save_nii(
            mask=vmask[0].permute(1, 2, 0).cpu().numpy().astype(np.int8),
            pred=test_output_argmax[0].permute(1, 2, 0).cpu().numpy().astype(np.int8),
            output_dir=os.path.join(output_path, str(case_id)),
            affine=affine,
            case_id=case_id
        )
        
        # 复制原始数据
        case_dir = test_df.loc[test_df['patient_idx'] == case_id, 'patient_dir'].values[0]
        for fname in os.listdir(case_dir):
            shutil.copy(os.path.join(case_dir, fname), os.path.join(output_path, str(case_id), fname))
            
    except Exception as e:
        print(f"案例 {case_id} 保存失败: {str(e)}")
        raise

def save_nii(
    mask: np.ndarray,
    pred: np.ndarray,
    output_dir: str,
    affine: np.ndarray,
    case_id: str
) -> None:
    """保存NIfTI文件"""
    os.makedirs(output_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, affine), os.path.join(output_dir, f'{case_id}_mask.nii.gz'))
    nib.save(nib.Nifti1Image(pred, affine), os.path.join(output_dir, f'{case_id}_pred.nii.gz'))

def process_metrics(raw_metrics: np.ndarray) -> Dict[str, Tuple[float, float, float, float]]:
    """处理指标数据"""
    return {
        'Dice': raw_metrics[0],
        'Jaccard': raw_metrics[1],
        'Precision': raw_metrics[2],
        'Recall': raw_metrics[3],
        'H95': raw_metrics[4]
    }

def generate_report(
    model_name: str,
    ckpt_path: str,
    inference_time: float,
    metrics: Dict[str, Tuple[float, float, float, float]],
    output_path: str
) -> None:
    """生成评估报告"""
    # 构建表格数据
    table_data = [
        [metric, *[f"{v:.4f}" for v in values]]
        for metric, values in metrics.items()
    ]
    
    # 生成报告内容
    report = f"""
╒══════════════════════════════════╕
│       医学影像推理报告            │
╘══════════════════════════════════╛
    
[模型信息]
模型名称: {model_name}
检查点路径: {ckpt_path}
推理耗时: {inference_time:.2f}秒
    
[评估指标]
{tabulate(table_data, headers=["指标", "MEAN", "ET", "TC", "WT"], tablefmt="fancy_grid")}
    
[输出目录]
{output_path}
"""
    
    # 保存报告
    with open(os.path.join(output_path, "report.txt"), 'w') as f:
        f.write(report)
    print(report)

def default_affine() -> np.ndarray:
    """生成默认仿射矩阵"""
    return np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 239],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def trans_from_wins_to_linux(path):
        path = path.replace('D:', '/mnt/d')
        path = path.replace('\\', '/')
        return path

if __name__ == '__main__':
    
    csv_file = '/root/workspace/BraTS_Solution/data/test.csv'
    
    out_dir = '/mnt/d/results/1_对比网络结果/output'
    os.makedirs(out_dir, exist_ok=True)
        
    model_names = ['UNet3D', 'AttentionUNet3D']
    test_df = pd.read_csv(csv_file)
    test_loader = load_data(csv_file, batch_size=1, num_workers=4, local_train=False, test_length=10)

    # 初始化配置
    unet_common_config = {
        'test_df': test_df,
        'test_loader': test_loader,
        'output_root': out_dir,
        'model': None,
        'metricer': EvaluationMetrics(),
        'scaler': GradScaler(),
        'optimizer': None,
        'ckpt_path': None
    }
    
    # ##! UNet3D
    model = load_model('UNet3D')
    unet_common_config.update(
        {
            'model':        model,
            'optimizer':    AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001),
            'ckpt_path':    trans_from_wins_to_linux(r"D:\results\1_对比网络结果\[√]UNet3D_2025-03-29_lr0.0001_mlr1e-06_Tmax100_100_100\checkpoints\UNet3D_final_model.pth")
        }
    )
    inference(**unet_common_config, stride_ratio=0)

    
    # ##! AttUNet3D
    model = load_model('AttUNet3D')
    unet_common_config.update(
        {
            'model':        model,
            'optimizer':    AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001),
            'ckpt_path':    trans_from_wins_to_linux(r"D:\results\1_对比网络结果\[√]AttUNet3D_2025-04-10_lr0.0001_mlr1e-06_Tmax100_100_100\checkpoints\AttUNet3D_final_model.pth")
        }
    )
    inference(**unet_common_config, stride_ratio=0)
    
    # ##! UNETR
    model = load_model('UNETR')
    unet_common_config.update(
        {
            'model':        model,
            'optimizer':    AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001),
            'ckpt_path':    trans_from_wins_to_linux(r"D:\results\1_对比网络结果\[√]UNETR_2025-04-09_lr0.0001_mlr1e-06_Tmax100_100_100\checkpoints\UNETR_final_model.pth")
        }
    )
    inference(**unet_common_config, stride_ratio=0)
    

    # ##! Mamba3D
    model = load_model('Mamba3d')
    unet_common_config.update(
        {
            'model':        model,
            'optimizer':    AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001),
            'ckpt_path':    trans_from_wins_to_linux(r"D:\results\1_对比网络结果\[√]Mamba3d_2025-04-01_lr0.0001_mlr1e-06_Tmax100_100_100\checkpoints\best_epoch85_loss0.1296_dice0.8713_20250401063106.pth")
        }
    )
    inference(**unet_common_config, stride_ratio=0)

    # ##! MogaNet
    model = load_model('MogaNet')
    unet_common_config.update(
        {
            'model':        model,
            'optimizer':    AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001),
            'ckpt_path':    trans_from_wins_to_linux(r"D:\results\1_对比网络结果\[√]MogaNet_2025-04-09_lr0.0001_mlr1e-06_Tmax100_100_100\checkpoints\MogaNet_final_model.pth")
        }
    )
    inference(**unet_common_config, stride_ratio=0)

    # ##! SegFormer3D
    model = load_model('SegFormer3D')
    unet_common_config.update(
        {
            'model':        model,
            'optimizer':    AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001),
            'ckpt_path':    trans_from_wins_to_linux(r"D:\results\1_对比网络结果\[√]SegFormer3D_2025-03-31_lr0.0001_mlr1e-06_Tmax100_100_100\checkpoints\SegFormer3D_final_model.pth")
        }
    )
    inference(**unet_common_config, stride_ratio=0)

    # # ##! DCLA_UNet_v2
    # model = load_model('DCLA_UNet_v2')
    # unet_common_config.update(
    #     {
    #         'model':        model,
    #         'optimizer':    AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001),
    #         'ckpt_path':    trans_from_wins_to_linux(r"D:\results\20250503_0.866\【0.866】DCLA_UNet_2025-05-03_lr0.0001_mlr1e-06_Tmax100_100_100\checkpoints\DCLA_UNet_final_model.pth")
    #     }
    # )
    # inference(**unet_common_config, stride_ratio=0)
    
    
    
    
    
    
    
    
    
    
    