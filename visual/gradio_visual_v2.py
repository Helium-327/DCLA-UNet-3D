import os
import csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import gradio as gr
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# 可视化配置
COLOR_MAP = ListedColormap([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], name='custom_discrete', N=4)
BOUNDARIES = [0, 1, 2, 3, 4]
NORM = BoundaryNorm(BOUNDARIES, COLOR_MAP.N)

class MultiModelVisualizer:
    def __init__(self):
        self.datasets_df: Optional[pd.DataFrame] = None
        self.model_dfs: Dict[str, pd.DataFrame] = {}
        self.color_palette = plt.cm.get_cmap('tab10')
        
    def _get_available_patients(self) -> List[str]:
        """获取所有模型共有的患者ID"""
        if not self.model_dfs:
            return []
        
        patient_ids = set()
        for df in self.model_dfs.values():
            patient_ids.update(df["PatientID"].unique())
        return sorted(patient_ids)
    
    def process_data(self, dataset_csv: gr.File, model_csvs: List[gr.File]) -> Tuple[bool, str]:
        """处理上传的CSV文件"""
        try:
            self.datasets_df = None
            self.model_dfs.clear()
            
            if not dataset_csv:
                return False, "请上传原始数据集CSV"
            self.datasets_df = pd.read_csv(dataset_csv.name)
            
            if not model_csvs:
                return False, "请至少上传一个模型预测CSV"
                
            for csv_file in model_csvs:
                if csv_file:
                    model_name = Path(csv_file.name).stem
                    self.model_dfs[model_name] = pd.read_csv(csv_file.name)
            
            def process_df(df):
                df = df.copy()
                df["PatientID"] = df["subDirName"].str.split('_').str[-1]
                df["ModalType"] = df["fileName"].str.split('.').str[0].str.split('_').str[-1]
                return df
                
            if self.datasets_df is not None:
                self.datasets_df = process_df(self.datasets_df)
                
            for model in self.model_dfs:
                self.model_dfs[model] = process_df(self.model_dfs[model])
                
            return True, f"成功加载 {len(self.model_dfs)} 个模型"
            
        except Exception as e:
            return False, f"数据处理失败: {str(e)}"

    def get_patient_data(self, patient_id: str) -> Dict[str, str]:
        """获取指定患者的数据路径"""
        if not patient_id:
            raise ValueError("未选择患者ID")
            
        data = {"original": "", "ground_truth": ""}
        
        try:
            if self.datasets_df is not None:
                original = self.datasets_df[
                    (self.datasets_df["PatientID"] == patient_id) & 
                    (self.datasets_df["ModalType"] == "t1")
                ]
                if not original.empty:
                    data["original"] = original["absPathOfFile"].iloc[0]
                
                gt = self.datasets_df[
                    (self.datasets_df["PatientID"] == patient_id) & 
                    (self.datasets_df["ModalType"] == "seg")
                ]
                if not gt.empty:
                    data["ground_truth"] = gt["absPathOfFile"].iloc[0]
        except Exception as e:
            print(f"获取基础数据错误: {str(e)}")
        
        for model_name in self.model_dfs:
            try:
                pred = self.model_dfs[model_name][
                    (self.model_dfs[model_name]["PatientID"] == patient_id) & 
                    (self.model_dfs[model_name]["ModalType"] == "pred")
                ]
                if not pred.empty:
                    data[model_name] = pred["absPathOfFile"].iloc[0]
            except Exception as e:
                print(f"获取{model_name}数据错误: {str(e)}")
        
        return data

    def plot_comparison(self, data_dict: Dict, slice_index: int, axis: int) -> plt.Figure:
        """纵向对比可视化"""
        if not data_dict.get("original"):
            raise gr.Error("缺少原始图像数据")
            
        try:
            # 加载基础数据
            original_img = nib.load(data_dict["original"]).get_fdata().transpose(2,0,1)
            max_slices = original_img.shape[axis]
            slice_index = min(max(slice_index, 0), max_slices-1)
            
            # 创建可视化布局
            rows = 1 + len(self.model_dfs)  # 基础行 + 模型行
            fig = plt.figure(figsize=(20, 5 * rows))
            grid = plt.GridSpec(rows, 5, hspace=0.1, wspace=0.1)
            
            # 基础行可视化
            base_slice = self._get_slice(original_img, slice_index, axis)
            
            # # 原图
            # ax_original = fig.add_subplot(grid[0, 0])
            # ax_original.imshow(base_slice, cmap='gray')
            # ax_original.set_title("Original Image", fontsize=10)
            # ax_original.axis('off')
            
            # GT覆盖图
            ax_gt_overlay = fig.add_subplot(grid[0, 1])
            ax_gt_overlay.imshow(base_slice, cmap='gray')
            if data_dict.get("ground_truth"):
                gt_img = nib.load(data_dict["ground_truth"]).get_fdata().transpose(2,0,1)
                gt_img[gt_img==4] = 3  # 将4标记转换为3
                gt_slice = self._get_slice(gt_img, slice_index, axis)
                ax_gt_overlay.imshow(np.ma.masked_where(gt_slice==0, gt_slice), 
                                   cmap=COLOR_MAP, norm=NORM)
            ax_gt_overlay.set_title("GT Overlay", fontsize=10)
            ax_gt_overlay.axis('off')
            
            # GT分解图
            def plot_gt_subtype(ax, title, mask_func):
                ax.imshow(base_slice, cmap='gray')
                if data_dict.get("ground_truth"):
                    # masked = np.ma.masked_where(~mask_func(gt_slice), gt_slice)
                    ax.imshow(mask_func(gt_slice), norm=NORM, cmap=COLOR_MAP)
                ax.set_title(title, fontsize=10)
                ax.axis('off')
            
            plot_gt_subtype(fig.add_subplot(grid[0, 2]), "GT-ET", lambda x: x == 3)
            plot_gt_subtype(fig.add_subplot(grid[0, 3]), "GT-TC", lambda x: (x == 1) | (x == 3))
            plot_gt_subtype(fig.add_subplot(grid[0, 4]), "GT-WT", lambda x: x >= 1)

            # 模型行可视化
            for row, (model_name, _) in enumerate(self.model_dfs.items(), 1):
                if not data_dict.get(model_name):
                    continue
                
                try:
                    pred_img = nib.load(data_dict[model_name]).get_fdata().transpose(2,0,1)
                    pred_slice = self._get_slice(pred_img, slice_index, axis)
                    
                    # 模型名称
                    ax_name = fig.add_subplot(grid[row, 0])
                    ax_name.text(0.5, 0.5, model_name, ha='center', va='center', 
                                fontsize=12, color=self.color_palette(row/len(self.model_dfs)))
                    ax_name.axis('off')
                    
                    # 预测覆盖图
                    ax_pred = fig.add_subplot(grid[row, 1])
                    ax_pred.imshow(base_slice, cmap='gray')
                    ax_pred.imshow(np.ma.masked_where(pred_slice==0, pred_slice), 
                                 norm=NORM, cmap=COLOR_MAP)
                    ax_pred.axis('off')
                    ax_pred.set_title("Prediction", fontsize=10)
                    
                    # 预测分解图
                    def plot_pred_subtype(ax, title, mask_func):
                        ax.imshow(base_slice, cmap='gray')
                        # masked = np.ma.masked_where(~mask_func(pred_slice), pred_slice)
                        ax.imshow(mask_func(pred_slice), norm=NORM, cmap=COLOR_MAP)
                        ax.set_title(title, fontsize=10)
                        ax.axis('off')
                    
                    plot_pred_subtype(fig.add_subplot(grid[row, 2]), "ET", lambda x: x == 3)
                    plot_pred_subtype(fig.add_subplot(grid[row, 3]), "TC", lambda x: (x == 1) | (x == 3))
                    plot_pred_subtype(fig.add_subplot(grid[row, 4]), "WT", lambda x: x >= 1)
                    
                except Exception as e:
                    print(f"处理{model_name}时出错: {str(e)}")
                    continue
            
            # 添加图例
            legend_elements = [
                Patch(facecolor=COLOR_MAP(1), label='TC (Label 1)'),
                Patch(facecolor=COLOR_MAP(2), label='WT (Label 2)'),
                Patch(facecolor=COLOR_MAP(3), label='ET (Label 3)')
            ]
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
            
            # 添加切片信息
            axis_names = ['Axial', 'Sagittal', 'Coronal']
            fig.suptitle(f"Slice: {slice_index} | View: {axis_names[axis]}", 
                        y=0.98, fontsize=12)
            
            # plt.tight_layout()
            return fig
            
        except Exception as e:
            raise gr.Error(f"可视化失败: {str(e)}")

    def _get_slice(self, volume: np.ndarray, idx: int, axis: int) -> np.ndarray:
        """获取三维体积的二维切片"""
        slicers = {
            0: lambda img, i: img[i, :, :],
            1: lambda img, i: img[:, i, :],
            2: lambda img, i: img[:, :, i]
        }
        return slicers[axis](volume, idx)

def create_gradio_interface():
    visualizer = MultiModelVisualizer()
    
    with gr.Blocks(title="BraTS 多模型对比系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# BraTS 多模型对比可视化系统")
        
        with gr.Row():
            dataset_csv = gr.File(label="原始数据集CSV", file_types=[".csv"])
            model_csvs = gr.Files(label="上传模型预测CSV（可多选）", 
                                file_types=[".csv"], 
                                file_count="multiple")
        
        with gr.Row():
            with gr.Column(scale=1):
                process_btn = gr.Button("加载数据", variant="primary")
                data_status = gr.Textbox(label="数据状态", interactive=False)
                patient_dropdown = gr.Dropdown(
                    label="选择患者ID", 
                    interactive=True,
                    allow_custom_value=False
                )
                
            with gr.Column(scale=2):
                slice_slider = gr.Slider(
                    minimum=0, maximum=200, value=80, step=1,
                    label="切片索引"
                )
                axis_radio = gr.Radio(
                    [0, 1, 2], value=0, 
                    label="显示轴向",
                    info="0:轴向 1:矢状 2:冠状"
                )
        
        comparison_plot = gr.Plot(label="多模型对比视图")
        
        def process_and_update(dataset, models):
            success, msg = visualizer.process_data(dataset, models)
            if success:
                return msg, gr.Dropdown(
                    choices=visualizer._get_available_patients(),
                    value=visualizer._get_available_patients()[0] if visualizer._get_available_patients() else None
                )
            raise gr.Error(msg)
            
        process_btn.click(
            fn=process_and_update,
            inputs=[dataset_csv, model_csvs],
            outputs=[data_status, patient_dropdown]
        )
        
        def update_visualization(patient_id, slice_idx, axis):
            if not patient_id:
                raise gr.Warning("请先选择患者ID")
            try:
                data = visualizer.get_patient_data(patient_id)
                return visualizer.plot_comparison(data, slice_idx, axis)
            except Exception as e:
                raise gr.Error(str(e))
        
        controls = [patient_dropdown, slice_slider, axis_radio]
        for control in controls:
            control.change(
                fn=update_visualization,
                inputs=controls,
                outputs=comparison_plot
            )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=True
    )