import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gradio as gr
import io
import base64
import matplotlib

# 设置matplotlib支持中文
try:
    # 尝试设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告：无法设置中文字体，图表中的中文可能无法正确显示")

# 函数：创建气泡图
def create_bubble_chart(df, x_col, y_col, size_col, main_model=None, title="Model Comparison Bubble Chart", base_bubble_size=300):
    """
    创建气泡图
    
    参数:
    df: pandas DataFrame，包含要绘制的数据
    x_col: x轴列名（参数量）
    y_col: y轴列名（指标名，如Dice）
    size_col: 气泡大小列名（FLOPs）
    main_model: 主模型名称，将被突出显示
    title: 图表标题
    base_bubble_size: 用于计算气泡大小的基础值，用户可通过Gradio调整
    
    返回:
    fig: matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=(10, 8))  # 调整图表大小，为底部图例留出空间
    
    # base_bubble_size 现在表示最大气泡的期望面积（在散点图中的 's' 值）
    max_size_val = df[size_col].max()
    
    # 直接将 base_bubble_size 作为最大气泡的面积
    target_max_area_for_scatter = base_bubble_size 
    
    # 计算原始 size_col 值到散点图所需面积的缩放因子
    # 这个因子将应用于 df[size_col] 以获得 's' 值（面积）
    scatter_area_scale_factor = target_max_area_for_scatter / max_size_val if max_size_val > 0 else 1
    
    sizes = df[size_col] * scatter_area_scale_factor # 这些是散点图的面积（s 值）
    
    # 为每个模型分配不同的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    
    # 创建散点图，每个模型使用不同颜色
    scatter_plots = []
    for i, (idx, row) in enumerate(df.iterrows()):
        # 如果是主模型，使用红色并增加大小
        if main_model and idx == main_model:
            scatter = ax.scatter(row[x_col], row[y_col], s=sizes.iloc[i], 
                        color='red', alpha=0.9, edgecolors='black', linewidths=2, zorder=10)
            # 添加辅助线突出主网络参数量和dice系数
            ax.axvline(x=row[x_col], color='red', linestyle='--', alpha=0.5, zorder=5)
            ax.axhline(y=row[y_col], color='red', linestyle='--', alpha=0.5, zorder=5)
        else:
            scatter = ax.scatter(row[x_col], row[y_col], s=sizes.iloc[i], 
                        color=colors[i], alpha=0.7, edgecolors='black', linewidths=1)
        scatter_plots.append(scatter)
    
    # 添加模型名称标签 - 调整位置避免与气泡重叠
    for i, txt in enumerate(df.index):
        # 计算标签位置，避免与气泡重叠
        # 根据气泡大小计算偏移量
        bubble_size = sizes.iloc[i]
        bubble_radius = np.sqrt(bubble_size / np.pi)
        # offset_x = bubble_radius + 5  # 水平偏移量，增加以确保不重叠
        offset_x = bubble_radius  # 水平偏移量，增加以确保不重叠
        offset_y = 0  # 垂直偏移量，设为0使标签在气泡右侧
        
        # 如果是主模型，使用加粗字体
        if main_model and txt == main_model:
            ax.annotate(txt, (df[x_col].iloc[i], df[y_col].iloc[i]),
                       xytext=(offset_x, offset_y), textcoords='offset points', 
                       fontweight='bold', fontsize=12, ha='left', va='center')
        else:
            ax.annotate(txt, (df[x_col].iloc[i], df[y_col].iloc[i]),
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       ha='left', va='center')
    
    # 设置图表标题和轴标签
    ax.set_title(title, fontsize=16, y=1.15) # 调整标题位置，使其在图例上方
    ax.set_xlabel(x_col, fontsize=14)
    ax.set_ylabel(y_col, fontsize=14)
    
    # 设置网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 使用科学计数法表示横坐标（参数量）- 使用10^n形式
    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # 强制使用科学计数法
    ax.xaxis.set_major_formatter(formatter)
    
    # 调整图表布局，为顶部模型图例留出空间
    # plt.subplots_adjust(bottom=0.1)  # 调整底部边界，为图例留出空间
    
    # 添加模型图例
    if main_model:
        # 创建自定义图例，突出显示主模型
        legend_elements = []
        for i, model_name in enumerate(df.index):
            if model_name == main_model:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor='red', markersize=10, 
                                               label=f"{model_name} (Ours)"))
            else:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor=colors[i], markersize=8, 
                                               label=model_name))
    else:
        # 创建普通图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=colors[i], markersize=8, 
                                   label=model_name) for i, model_name in enumerate(df.index)]
    
    # 计算模型图例应该使用的列数，确保布局平衡
    model_ncol = min(6, len(legend_elements))
    
    # 添加模型图例在标题下方
    model_legend = ax.legend(handles=legend_elements, loc='lower center', 
                           bbox_to_anchor=(0.5, 1.02), title="Models", 
                           frameon=False, ncol=model_ncol, fontsize=9)
    ax.add_artist(model_legend)  # 保留这个图例
    
    # 设置图例标题为粗体
    plt.setp(model_legend.get_title(), fontweight='bold')

    # 创建气泡大小图例 - 使用实际数据中的气泡大小
    # 选择3个代表性的大小值
    size_values = [df[size_col].min(), 
                   (df[size_col].min() + df[size_col].max()) / 2, 
                   df[size_col].max()]
    
    size_handles = []
    size_labels = [f"{val:.1f}" for val in size_values]
    
    # 确保图例气泡大小与实际模型气泡完全一致
    for size_val in size_values:
        # 直接使用与散点图完全相同的大小计算方式
        # size_val 是原始的FLOPs值，乘以 scatter_area_scale_factor 得到散点图的 's' 值（面积）
        scatter_s_value = size_val * scatter_area_scale_factor
        # markersize 是直径，所以需要从面积反推直径
        # 面积 = pi * (直径/2)^2 => 直径 = 2 * sqrt(面积 / pi)
        legend_marker_diameter = 2 * np.sqrt(scatter_s_value / np.pi)
        size_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='gray', alpha=0.7,
                                      markersize=legend_marker_diameter,
                                      label=f"{size_val:.1f}"))
    
    # 将气泡大小图例放在图表内部右下角
    size_legend = ax.legend(handles=size_handles, labels=size_labels, 
                          loc='lower right', bbox_to_anchor=(0.98, 0.02), 
                          title=size_col, frameon=False, fontsize=8)
    
    # 确保图表布局合适
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # 调整紧凑布局，为顶部图例和标题留出空间
    return fig

# 示例数据
def get_example_data():
    # 创建示例数据
    data = {
        'Model': ['CLIP', 'ViT-L', 'ResNet50', 'EfficientNet', 'Swin-T', 'DeiT'],
        'FLOPs(G)': [17.6, 19.1, 4.1, 0.4, 4.5, 4.6],
        'Params(M)': [428.3, 307.0, 25.6, 5.3, 28.3, 22.1],
        'Dice': [0.852, 0.878, 0.761, 0.771, 0.813, 0.798]
    }
    df = pd.DataFrame(data)
    df.set_index('Model', inplace=True)
    return df

# 将matplotlib图形转换为图像
def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    # 转换为PIL Image对象，这是Gradio支持的格式
    from PIL import Image
    img = Image.open(buf)
    plt.close(fig)
    return img

# 手动输入模型数据创建气泡图
def create_chart_from_input(model_names, flops, params, dice, main_model=None, y_label="Dice", base_bubble_size=300):
    # 检查输入是否有效
    if not model_names or not flops or not params or not dice:
        return None, "请输入模型名称、FLOPs、参数量和指标值"
    
    # 分割输入字符串
    models = [x.strip() for x in model_names.split(',')]
    flops_values = [float(x.strip()) for x in flops.split(',')]
    params_values = [float(x.strip()) for x in params.split(',')]
    dice_values = [float(x.strip()) for x in dice.split(',')]
    
    # 检查长度是否匹配
    if len(models) != len(flops_values) or len(models) != len(params_values) or len(models) != len(dice_values):
        return None, f"输入的模型名称、FLOPs、参数量和{y_label}值数量不匹配"
    
    # 检查主模型是否在模型列表中
    if main_model and main_model not in models:
        return None, f"主模型 '{main_model}' 不在模型列表中"
    
    # 创建DataFrame
    data = {
        'Model': models,
        'FLOPs(G)': flops_values,
        'Params(M)': params_values,
        y_label: dice_values  # 使用动态的指标名
    }
    
    df = pd.DataFrame(data)
    df.set_index('Model', inplace=True)
    
    # 创建自定义标题
    title = f"Model Comparison: {y_label} vs Params(M) (Size: FLOPs(G))"
    
    # 创建气泡图 - 横坐标为参数量，纵坐标为指定指标，气泡大小表示FLOPs
    fig = create_bubble_chart(df, 'Params(M)', y_label, 'FLOPs(G)', main_model, title=title, base_bubble_size=base_bubble_size)
    
    # 将matplotlib图形转换为图像数据
    img_data = fig_to_image(fig)
    
    return img_data, "气泡图创建成功！"

# 从CSV文件创建气泡图
def create_chart_from_csv(csv_file, x_col, y_col, size_col, main_model=None, base_bubble_size=300):
    try:
        # 读取CSV文件
        # Gradio 3.x 版本中，csv_file 是一个 NamedString 对象，其 name 属性是文件路径
        df = pd.read_csv(csv_file.name)
        
        # 如果用户没有指定y_col，则使用最后一列作为y_col
        if not y_col or y_col.strip() == "":
            y_col = df.columns[-1]
            print(f"未指定纵坐标列名，使用最后一列 '{y_col}' 作为纵坐标")
        
        # 检查必要的列是否存在
        required_cols = [x_col, y_col, size_col]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f"CSV文件中缺少以下列: {', '.join(missing_cols)}"
        
        # 如果有'Model'或'模型'列，将其设置为索引
        if 'Model' in df.columns:
            df.set_index('Model', inplace=True)
        elif '模型' in df.columns:
            df.set_index('模型', inplace=True)
        elif 'model' in df.columns:
            df.set_index('model', inplace=True)
        else:
            # 如果没有明确的模型列，使用第一列作为索引
            df.set_index(df.columns[0], inplace=True)
            print(f"使用第一列 '{df.index.name}' 作为模型名称")
        
        # 检查主模型是否在模型列表中
        if main_model and main_model not in df.index:
            return None, f"主模型 '{main_model}' 不在CSV文件的模型列表中"
        
        # 创建自定义标题，包含指标信息
        title = f"Model Comparison: {y_col} vs {x_col} (Size: {size_col})"
        
        # 创建气泡图
        fig = create_bubble_chart(df, x_col, y_col, size_col, main_model, title=title, base_bubble_size=base_bubble_size)
        # 将matplotlib图形转换为图像数据
        img_data = fig_to_image(fig)
        return img_data, "气泡图创建成功！"
    
    except Exception as e:
        return None, f"创建气泡图时出错: {str(e)}"

# 创建Gradio界面
def create_gradio_interface():
    # 示例数据
    example_df = get_example_data()
    
    # 手动输入选项卡
    with gr.Blocks() as manual_input_tab:
        gr.Markdown("## 通过手动输入创建气泡图")
        with gr.Row():
            with gr.Column():
                model_names = gr.Textbox(label="模型名称（用逗号分隔）", placeholder="CLIP, ViT-L, ResNet50")
                flops = gr.Textbox(label="FLOPs(G)（用逗号分隔）", placeholder="17.6, 19.1, 4.1")
                params = gr.Textbox(label="参数量(M)（用逗号分隔）", placeholder="428.3, 307.0, 25.6")
                y_label = gr.Textbox(label="纵坐标指标名称", placeholder="Dice", value="Dice")
                dice = gr.Textbox(label="指标值（用逗号分隔）", placeholder="0.852, 0.878, 0.761")
                main_model = gr.Textbox(label="主模型名称（将被突出显示）", placeholder="输入上述模型列表中的一个名称")
                base_bubble_size_slider = gr.Slider(minimum=100, maximum=1000, value=300, label="基础气泡大小", info="调整气泡的整体大小")
                create_btn = gr.Button("创建气泡图")
            
            with gr.Column():
                output_image = gr.Image(label="生成的气泡图")
                output_text = gr.Textbox(label="状态信息")
        
        # 设置示例输入
        example_models = ", ".join(example_df.index.tolist())
        example_flops = ", ".join([str(x) for x in example_df["FLOPs(G)"].tolist()])
        example_params = ", ".join([str(x) for x in example_df["Params(M)"].tolist()])
        example_dice = ", ".join([str(x) for x in example_df["Dice"].tolist()])
        example_main_model = example_df.index[0]  # 默认使用第一个模型作为主模型
        example_y_label = "Dice"  # 默认纵坐标指标名
        
        example_btn = gr.Button("加载示例数据")
        example_btn.click(
            lambda: (example_models, example_flops, example_params, example_y_label, example_dice, example_main_model, 300),
            outputs=[model_names, flops, params, y_label, dice, main_model, base_bubble_size_slider]
        )
        
        # 设置创建按钮的点击事件
        create_btn.click(
            create_chart_from_input,
            inputs=[model_names, flops, params, dice, main_model, y_label, base_bubble_size_slider],
            outputs=[output_image, output_text]
        )
    
    # CSV上传选项卡
    with gr.Blocks() as csv_upload_tab:
        gr.Markdown("## 通过上传CSV文件创建气泡图")
        with gr.Row():
            with gr.Column():
                csv_file = gr.File(label="上传CSV文件", file_types=[".csv"])
                x_col = gr.Textbox(label="X轴列名（参数量）", value="Params(M)")
                y_col = gr.Textbox(label="Y轴列名（指标名）", value="Dice")
                size_col = gr.Textbox(label="气泡大小列名（FLOPs）", value="FLOPs(G)")
                main_model_csv = gr.Textbox(label="主模型名称（将被突出显示）", placeholder="输入CSV中的一个模型名称")
                base_bubble_size_slider_csv = gr.Slider(minimum=100, maximum=1000, value=300, label="基础气泡大小", info="调整气泡的整体大小")
                upload_btn = gr.Button("创建气泡图")
            
            with gr.Column():
                csv_output_image = gr.Image(label="生成的气泡图")
                csv_output_text = gr.Textbox(label="状态信息")
        
        # 设置上传按钮的点击事件
        def process_upload(file_obj, x, y, size, main_model, base_bubble_size):
            if file_obj is None:
                return None, "请上传CSV文件"
            # 确保返回的是图像数据而不是matplotlib图形对象
            result = create_chart_from_csv(file_obj, x, y, size, main_model if main_model else None, base_bubble_size)
            return result
        
        # 添加CSV示例说明
        gr.Markdown("""
        ### CSV文件格式说明
        CSV文件应包含以下列（列名可自定义）：
        - 第一列：模型名称（将自动设为索引）
        - 参数量列：模型参数量，单位为M（默认列名：Params(M)）
        - 指标列：模型性能指标，如Dice、Accuracy等（默认列名：Dice）
        - FLOPs列：模型计算量，单位为G（默认列名：FLOPs(G)）
        
        示例：
        ```
        Model,FLOPs(G),Params(M),Dice
        CLIP,17.6,428.3,0.852
        ViT-L,19.1,307.0,0.878
        ResNet50,4.1,25.6,0.761
        ```
        """)
        
        upload_btn.click(
            process_upload,
            inputs=[csv_file, x_col, y_col, size_col, main_model_csv, base_bubble_size_slider_csv],
            outputs=[csv_output_image, csv_output_text]
        )
    
    # 创建选项卡界面
    demo = gr.TabbedInterface(
        [manual_input_tab, csv_upload_tab],
        ["手动输入", "CSV上传"]
    )
    
    return demo

# 主函数
if __name__ == "__main__":
    try:
        print("正在初始化Gradio界面...")
        # 创建Gradio界面
        demo = create_gradio_interface()
        
        print("正在启动Gradio应用...")
        # 启动Gradio应用，设置share=True以便在Colab或远程服务器上使用
        demo.launch(share=True)
        print("Gradio应用已成功启动！")
    except Exception as e:
        print(f"启动Gradio应用时出错: {str(e)}")
        import traceback
        traceback.print_exc()