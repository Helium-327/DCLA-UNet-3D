import os
import copy

from pathlib import Path # 路径操作
import pandas as pd
import nibabel as nib
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, IntSlider
from IPython.display import display, clear_output
from matplotlib.colors import ListedColormap, BoundaryNorm
import tracemalloc


COLOR_MAP = ListedColormap([(0,0,0), (1,0,0), (0,1,0), (0,0,1)], name='custom_discrete', N=4)
BOUNDARIES = [0, 1, 2, 3, 4]
NORM = BoundaryNorm(BOUNDARIES, COLOR_MAP.N)

import os
import csv
from pathlib import Path
from tqdm import tqdm

def generate_csv(dirInput, csvPath, rowNameList):
    """生成CSV数据
    Args:
        dirInput: 输入路径
        csvPath: 输出CSV文件路径
        rowNameList: 列名列表
    Returns:
        data: 包含目录结构的列表，每个元素是一个子列表，包含子目录名、绝对路径、文件名和文件绝对路径
    """
    dirInput = Path(dirInput)
    dataList = []
    
    # 获取所有子目录
    subdirs = [d for d in dirInput.iterdir() if d.is_dir()]
    
    # 主进度条
    for subdir in tqdm(subdirs, desc="Processing directories", unit="dir"):
        subdir_name = subdir.name
        subdir_abs_path = str(subdir.resolve())
        
        # 获取当前子目录下的所有文件
        files = [f for f in subdir.iterdir() if f.is_file()]
        
        # 子目录文件处理进度条
        for file_item in tqdm(files, desc=f"Files in {subdir_name}", unit="file", leave=False):
            file_name = file_item.name
            file_abs_path = str(file_item.resolve())
            
            # 添加到数据列表
            dataList.append([
                subdir_name,
                subdir_abs_path,
                file_name,
                file_abs_path
            ])
            
        # 写入CSV文件
    with open(csvPath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(rowNameList)
        # 写入数据
        writer.writerows(dataList)
    
    print(f"成功生成CSV文件：{csvPath}")

# 添加modal列
def add_modal_type(filename):
    return filename.split('.')[0].split('_')[-1]
def add_patient_id(dirName):
    return dirName.split('_')[-1]  # 假设ID在第二个位置

def process_pd(df: pd.DataFrame):
    # 添加modal列和patient_id列
    ## 添加PatientID列
    df["PatientID"] = df["subDirName"].apply(add_patient_id) 
    ## 添加ModalType列
    df["ModalType"] = df["fileName"].apply(add_modal_type)   
    return df

def convert_path(path: str) -> str:
    """
    将Windows路径转换为WSL/Linux路径
    支持所有盘符转换 (D: -> /mnt/d, E: -> /mnt/e)
    自动标准化路径格式
    """
    # 将路径转换为Path对象自动处理不同OS的分隔符
    p = str(Path(path)) 
    
    # 处理盘符转换 (如 D: -> /mnt/d)
    if ':' in p:  # 仅处理包含盘符的路径
        drive, rest = p.split(':')
        # print(rest)
        rest = str(rest).replace('\\', '/').split('/')  # 替换反斜杠并分割路径
        # print(rest)
        linux_path = f"/mnt/{drive.lower()}/{'/'.join(rest)}"
        p = Path(linux_path)
        
    return str(p) # 返回绝对路径


def multi_generate_csv(model_config: dict):
    """
    批量生成模型预测的CSV文件
    Args:
        model_config: 包含模型名称和对应路径的字典
    """
    for model_name, dirInput in model_config.items():
        csvPath = Path(dirInput).parent / f'{model_name}.csv'  # 确保目录存在
        print(f"正在生成 {model_name} 的CSV文件...")
        generate_csv(
            dirInput=dirInput,
            csvPath=csvPath,
            rowNameList=['subDirName', 'absPathOfSubDir', 'fileName', 'absPathOfFile']
        )

if __name__ == '__main__':
       
    model_config = {
    # "UNet3D":                    convert_path(r"D:\results\DCLA_Unet_final\【0.836】UNet3D_2025-03-29_lr0.0001_mlr1e-06_Tmax100_100_100\output\UNet3D_20250605150443"),
    # "UNETR":                     convert_path(r"D:\results\DCLA_Unet_final\【0.855】UNETR_2025-04-09_lr0.0001_mlr1e-06_Tmax100_100_100\output\UNETR_20250605152256"),
    # "AttnUNet3D":                convert_path(r"D:\results\DCLA_Unet_final\【0.86】AttUNet3D_2025-05-28_lr0.0003_mlr1e-06_Tmax100_100_100\output\AttUNet3D_20250605151326"),
    # "UNETR_PP":                  convert_path(r"D:\results\DCLA_Unet_final\【0.867】UNETR_PP_2025-05-29_lr0.0003_mlr1e-06_Tmax100_100_100\output\UNETR_PP_20250605153158"),
    # "MogaNet":                   convert_path(r"D:\results\DCLA_Unet_final\【0.859】MogaNet_2025-04-09_lr0.0001_mlr1e-06_Tmax100_100_100\output\MogaNet_20250605154939"),
    # "Mamba3D":                   convert_path(r"D:\results\DCLA_Unet_final\【0.816】Mamba3d_2025-05-15_lr0.0001_mlr1e-06_Tmax100_100_100\output\Mamba3d_20250605154047"),
    # "SegFormer3D":               convert_path(r"D:\results\DCLA_Unet_final\【0.844】SegFormer3D_2025-05-30_lr0.0003_mlr1e-06_Tmax100_100_100\output\SegFormer3D_20250605155900"),
    "DCLA_UNet_final":           convert_path(r"D:\results\DCLA_Unet_final\【0.88】DCLA_UNet_final_2025-05-27_lr0.0003_mlr1e-06_Tmax100_100_100\output\DCLA_UNet_final_20250605185736"),
    "BaseLine_S_SLK_final":      convert_path(r"D:\results\DCLA_Unet_final\【0.833】BaseLine_S_SLK_final_2025-05-30_lr0.0003_mlr1e-06_Tmax100_100_100\output\BaseLine_S_SLK_final_20250605191059"),
    "ResUNetBaseline_S":         convert_path(r"D:\results\DCLA_Unet_final\【0.869】ResUNetBaseline_S_2025-05-29_lr0.0003_mlr1e-06_Tmax100_100_100\output\ResUNetBaseline_S_20250605160709"),
    "DCLA_UNet_withoutDCLA_v2_6": convert_path(r"D:\results\DCLA_Unet_final\【0.872】DCLA_UNet_withoutDCLA_v2_6_2025-06-03_lr0.0003_mlr1e-06_Tmax100_100_100\output\DCLA_UNet_withoutDCLA_v2_6_20250605165651"),
    "BaseLine_S_DCLA_SLK_final": convert_path(r"D:\results\DCLA_Unet_final\【0.875】BaseLine_S_DCLA_SLK_final_2025-05-30_lr0.0003_mlr1e-06_Tmax100_100_100\output\BaseLine_S_DCLA_SLK_final_20250605192059"),
    "BaseLine_S_DCLA_final":     convert_path(r"D:\results\DCLA_Unet_final\【0.876】BaseLine_S_DCLA_final_2025-05-31_lr0.0003_mlr1e-06_Tmax100_100_100\output\BaseLine_S_DCLA_final_20250605170750"),
    "BaseLine_S_SLK_MSF_final": convert_path(r"D:\results\DCLA_Unet_final\【0.881】BaseLine_S_SLK_MSF_final_2025-05-29_lr0.0003_mlr1e-06_Tmax100_100_100\output\BaseLine_S_SLK_MSF_final_20250605195505"),
    "BaseLine_S_MSF_final":      convert_path(r"D:\results\DCLA_Unet_final\【0.884】BaseLine_S_MSF_final_2025-05-30_lr0.0003_mlr1e-06_Tmax100_100_100\output\BaseLine_S_MSF_final_20250605193112"),
    "BaseLine_S_DCLA_MSF_final":  convert_path(r"D:\results\DCLA_Unet_final\【0.885】BaseLine_S_DCLA_MSF_final_2025-05-30_lr0.0003_mlr1e-06_Tmax100_100_100\output\BaseLine_S_DCLA_MSF_final_20250605194313")
    }
    multi_generate_csv(model_config)


