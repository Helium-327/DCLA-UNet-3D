import os
import sys
import csv
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import gradio as gr
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import ListedColormap, BoundaryNorm

# 颜色映射配置
COLOR_MAP = ListedColormap([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], name='custom_discrete', N=4)
BOUNDARIES = [0, 1, 2, 3, 4]
NORM = BoundaryNorm(BOUNDARIES, COLOR_MAP.N)

class BraTSVisualizer:
    def __init__(self):
        self.datasets_df: Optional[pd.DataFrame] = None
        self.pred_df: Optional[pd.DataFrame] = None
        self.modal_types: List[str] = ['t1', 't1ce', 't2', 'flair']
        
    def convert_path(self, path: str) -> str:
        """路径转换器"""
        if path is None:
            return ""
        p = str(Path(path))
        if ':' in p:
            drive, rest = p.split(':')
            rest = str(rest).replace('\\', '/').split('/')
            linux_path = f"/mnt/{drive.lower()}/{'/'.join(rest)}"
            p = Path(linux_path)
        return str(p)

    def generate_csv(self, dir_input: str, csv_path: str, row_names: List[str]) -> pd.DataFrame:
        """生成CSV文件"""
        dir_input = Path(dir_input)
        data_list = []
        
        subdirs = [d for d in dir_input.iterdir() if d.is_dir()]
        
        for subdir in tqdm(subdirs, desc="Processing directories", unit="dir"):
            subdir_name = subdir.name
            subdir_abs_path = str(subdir.resolve())
            
            files = [f for f in subdir.iterdir() if f.is_file()]
            
            for file_item in tqdm(files, desc=f"Files in {subdir_name}", unit="file", leave=False):
                file_name = file_item.name
                file_abs_path = str(file_item.resolve())
                
                data_list.append([
                    subdir_name, subdir_abs_path, file_name, file_abs_path
                ])
                
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_names)
            writer.writerows(data_list)
            
        print(f"成功生成CSV文件：{csv_path}")
        return pd.read_csv(csv_path)

    def process_data(self, dataset_csv: gr.File, pred_csv: gr.File) -> Tuple[bool, str]:
        """处理CSV数据"""
        try:
            # 处理上传的文件对象
            dataset_path = dataset_csv.name if hasattr(dataset_csv, 'name') else dataset_csv
            pred_path = pred_csv.name if hasattr(pred_csv, 'name') else pred_csv
            
            self.datasets_df = pd.read_csv(dataset_path)
            self.pred_df = pd.read_csv(pred_path)
            
            # 添加辅助列
            self.datasets_df["PatientID"] = self.datasets_df["subDirName"].apply(
                lambda x: x.split('_')[-1] if isinstance(x, str) else '')
            self.pred_df["PatientID"] = self.pred_df["subDirName"].apply(
                lambda x: x.split('_')[-1] if isinstance(x, str) else '')
                
            self.datasets_df["ModalType"] = self.datasets_df["fileName"].apply(
                lambda x: x.split('.')[0].split('_')[-1] if isinstance(x, str) else '')
            self.pred_df["ModalType"] = self.pred_df["fileName"].apply(
                lambda x: x.split('.')[0].split('_')[-1] if isinstance(x, str) else '')
                
            # 获取预测病例ID
            mask_ids = set(self.pred_df[self.pred_df["ModalType"] == "pred"]["PatientID"].unique())
            
            # 标记测试集
            conditions = self.datasets_df["PatientID"].isin(mask_ids) & (self.datasets_df["ModalType"] == "t1")
            self.datasets_df['setOfDatasets'] = None
            self.datasets_df.loc[conditions, "setOfDatasets"] = "test"
            
            return True, "数据处理成功"
        except Exception as e:
            return False, f"数据处理失败：{str(e)}"

    def get_patient_data(self, patient_id: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        """获取患者数据"""
        if self.datasets_df is None or self.pred_df is None:
            return None, "请先上传CSV文件"
            
        if not patient_id or not isinstance(patient_id, str):
            return None, "无效的患者ID"
            
        try:
            # 获取原始数据
            original_data = self.datasets_df[
                (self.datasets_df["PatientID"] == patient_id) & 
                (self.datasets_df["ModalType"] == "t1")
            ]
            if original_data.empty:
                return None, f"未找到患者ID {patient_id} 的原始数据"
            original_path = original_data["absPathOfFile"].iloc[0]
            
            # 获取预测数据
            pred_data = self.pred_df[
                (self.pred_df["PatientID"] == patient_id) & 
                (self.pred_df["ModalType"] == "pred")
            ]
            
            if pred_data.empty:
                return None, f"未找到患者ID {patient_id} 的预测数据"
            pred_path = pred_data["absPathOfFile"].iloc[0]
            
            # 获取GT数据
            gt_data = self.pred_df[
                (self.pred_df["PatientID"] == patient_id) & 
                (self.pred_df["ModalType"] == "mask")
            ]
            if gt_data.empty:
                return None, f"未找到患者ID {patient_id} 的GT数据"
            gt_path = gt_data["absPathOfFile"].iloc[0]
            
            return {
                "original": original_path,
                "prediction": pred_path,
                "ground_truth": gt_path
            }, None
            
        except Exception as e:
            return None, f"获取数据失败：{str(e)}"

    def plotter(self, original_path: str, pred_path: str, gt_path: str, 
               slice_index: int = 80, axis: int = 0) -> plt.Figure:
        """可视化核心函数"""
        try:
            # Load and validate images
            orginal_img = nib.load(original_path).get_fdata().transpose(2, 0, 1)
            pred_img = nib.load(pred_path).get_fdata().transpose(2, 0, 1)
            gt_img = nib.load(gt_path).get_fdata().transpose(2, 0, 1)
            
            # 数据校验
            if not (orginal_img.shape == pred_img.shape == gt_img.shape):
                raise ValueError("图像尺寸不匹配")
                
            if axis not in [0, 1, 2]:
                raise ValueError("轴向无效")
                
            max_slice = orginal_img.shape[axis] - 1
            slice_index = max(0, min(slice_index, max_slice))
            
            # 切片函数
            slicer = {
                0: lambda img, idx: img[idx, :, :],
                1: lambda img, idx: img[:, idx, :],
                2: lambda img, idx: img[:, :, idx]
            }
            
            original_slice = slicer[axis](orginal_img, slice_index)
            pred_slice = slicer[axis](pred_img, slice_index)
            gt_slice = slicer[axis](gt_img, slice_index)
            
            # 创建可视化数据
            fig, axes = plt.subplots(1, 5, figsize=(20, 8))
            
            # Ground Truth
            axes[0].imshow(original_slice, cmap='gray')
            original_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
            axes[0].imshow(original_gt, cmap=COLOR_MAP, norm=NORM, alpha=0.5)
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            
            # Prediction
            axes[1].imshow(original_slice, cmap='gray')
            original_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
            axes[1].imshow(original_pred, cmap=COLOR_MAP, norm=NORM, alpha=0.5)
            axes[1].set_title('Prediction')
            axes[1].axis('off')
            
            # Enhanced Tumor Prediction
            et_data = np.zeros_like(original_slice, dtype=np.int8)
            pred_et_slice = np.where(pred_slice == 3, 1, et_data)
            axes[2].imshow(pred_et_slice, cmap='gray')
            axes[2].set_title('Enhanced Tumor Prediction')
            axes[2].axis('off')
            
            # Tumor Core Prediction
            tc_data = np.zeros_like(original_slice, dtype=np.int8)
            pred_tc_slice = np.where((pred_slice == 3) | (pred_slice == 1), 1, tc_data)
            axes[3].imshow(pred_tc_slice, cmap='gray')
            axes[3].set_title('Tumor Core Prediction')
            axes[3].axis('off')
            
            # Whole Tumor Prediction
            wt_data = np.zeros_like(original_slice, dtype=np.int8)
            pred_wt_slice = np.where((pred_slice == 3) | (pred_slice == 2) | (pred_slice == 1), 1, wt_data)
            axes[4].imshow(pred_wt_slice, cmap='gray')
            axes[4].set_title('Whole Tumor Prediction')
            axes[4].axis('off')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"可视化错误: {str(e)}")
            raise gr.Error(f"可视化失败：{str(e)}")

def create_gradio_interface():
    visualizer = BraTSVisualizer()
    
    def process_csv(dataset_csv: gr.File, pred_csv: gr.File) -> Tuple[str, gr.Dropdown, Optional[Dict]]:
        """处理CSV上传"""
        if dataset_csv is None or pred_csv is None:
            raise gr.Error("请先上传两个CSV文件")
            
        status, msg = visualizer.process_data(dataset_csv, pred_csv)
        if status:
            # 获取所有患者ID
            patient_ids = list(visualizer.pred_df[visualizer.pred_df["ModalType"] == "pred"]["PatientID"].unique())
            return (
                msg, 
                gr.Dropdown(choices=patient_ids, value=patient_ids[0] if patient_ids else None),
                {"status": "success", "patients": len(patient_ids)}
            )
        else:
            raise gr.Error(msg)

    def update_visualization(patient_id: str, slice_index: int, axis: int) -> Tuple[str, plt.Figure]:
        """更新可视化"""
        if not patient_id:
            raise gr.Warning("请先选择患者ID")
            
        paths, error = visualizer.get_patient_data(patient_id)
        if error:
            raise gr.Error(error)
            
        fig = visualizer.plotter(
            paths["original"], 
            paths["prediction"], 
            paths["ground_truth"], 
            slice_index, 
            axis
        )
        
        return "可视化更新成功", fig

    def update_params(patient_id: str, slice_index: int, axis: int) -> List[List[str]]:
        """更新参数显示"""
        return [
            ["Patient ID", patient_id or "未选择"],
            ["Slice Index", str(slice_index)],
            ["Axis", str(axis)],
            ["Status", "Ready" if patient_id else "等待输入"]
        ]

    with gr.Blocks(title="BraTS Visualizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# BraTS 2021 多模态可视化系统")
        
        with gr.Row():
            # 路径参数设置块
            with gr.Column(scale=1):
                gr.Markdown("### 路径参数设置")
                with gr.Group():
                    dataset_csv = gr.File(label="上传数据集CSV", file_types=[".csv"])
                    pred_csv = gr.File(label="上传预测结果CSV", file_types=[".csv"])
                process_btn = gr.Button("处理数据", variant="primary")
                data_info = gr.Textbox(label="数据处理状态", interactive=False)
                patient_id_dropdown = gr.Dropdown(
                    label="选择患者ID",
                    interactive=True,
                    allow_custom_value=False
                )
                data_json = gr.JSON(label="数据统计")
            
            # 控件参数设置块和可视化块
            with gr.Column(scale=2):
                gr.Markdown("### 可视化控制")
                with gr.Row():
                    slice_slider = gr.Slider(
                        minimum=0, maximum=154, value=80, step=1, 
                        label="切片索引", interactive=True
                    )
                    axis_slider = gr.Slider(
                        minimum=0, maximum=2, value=0, step=1,
                        label="轴向", interactive=True
                    )
                
                visualize_btn = gr.Button("更新可视化", variant="primary")
                visualize_status = gr.Textbox(label="可视化状态", interactive=False)
                
                # 可视化块
                visualization = gr.Plot(label="三维脑肿瘤分割可视化")
        
        # 数据参数列表块
        gr.Markdown("### 数据参数列表")
        data_params = gr.Dataframe(
            headers=["参数", "值"], 
            label="当前配置参数",
            interactive=False,
            datatype=["str", "str"]
        )
        
        # 事件绑定
        process_btn.click(
            fn=process_csv,
            inputs=[dataset_csv, pred_csv],
            outputs=[data_info, patient_id_dropdown, data_json]
        )
        
        visualize_btn.click(
            fn=update_visualization,
            inputs=[patient_id_dropdown, slice_slider, axis_slider],
            outputs=[visualize_status, visualization]
        )
        
        # 参数显示更新
        patient_id_dropdown.change(
            fn=update_params,
            inputs=[patient_id_dropdown, slice_slider, axis_slider],
            outputs=data_params
        )
        
        slice_slider.change(
            fn=update_params,
            inputs=[patient_id_dropdown, slice_slider, axis_slider],
            outputs=data_params
        )
        
        axis_slider.change(
            fn=update_params,
            inputs=[patient_id_dropdown, slice_slider, axis_slider],
            outputs=data_params
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=None,
        inbrowser=True
    )