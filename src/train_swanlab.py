# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/07/23 15:28:23
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 训练流程
=================================================
'''

import os
import time
import shutil
import torch
import pandas as pd
import logging
import readline  # 解决input()无法使用Backspace的问题, ⚠️不能删掉
from datetime import datetime

from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter 

# 添加SwanLab导入
try:
    import swanlab
except ImportError:
    swanlab = None

from optimized_inference import inference
from train_and_val import train_one_epoch, val_one_epoch

from utils.ckpt_tools import *
from utils.logger_tools import *
from utils.shell_tools import *

from torchinfo import summary

# 复现模式（测试/论文）
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# # 性能模式（默认推荐）
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

# constant
TB_PORT = 6007
RANDOM_SEED = 42
scheduler_start_epoch = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 加速
# torch.backends.cudnn.benchmark = True        #! 加速固定输入/网络结构的训练，但需避免动态变化场景，如数据增强
# torch.backends.cudnn.deterministic = True     #! 确保结果可复现，但可能降低性能并引发兼容性问题

# 调试工具
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
# torch.autograd.set_detect_anomaly(True)     #! 检测梯度异常，但会降低性能（谨慎使用，哥们）

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)                 #让显卡产生的随机数一致

# logging.basicConfig(level=logging.INFO, format=' %(levelname)s - %(message)s- %(asctime)s - %(name)s')
# logging.getLogger().setLevel(logging.INFO)

# 定义日志格式
# def setup_logging(log_file=None):
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S',
#         filename=log_file,
#         filemode='a'
#     )

#     # 如果没有指定日志文件，也输出到控制台
#     if not log_file:
#         console = logging.StreamHandler()
#         console.setLevel(logging.INFO)
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         console.setFormatter(formatter)
#         logging.getLogger('').addHandler(console)

def get_current_time():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def format_value(value, decimals=3):
    return f"{value:.{decimals}f}"

def train(model, Metrics, train_loader,  val_loader, test_loader, scaler, optimizer, scheduler, loss_function, 
          num_epochs, device, results_dir, logs_path, output_path, start_epoch, best_val_loss, test_csv,
          tb=False,  
          interval=10, 
          save_max=10, 
          early_stopping_patience=10,
          resume_tb_path=None,
          # SwanLab相关参数
          slb=False,
          slb_project=None,
          slb_experiment="default_exp",
          slb_config=None,
          logger=None
          ):
    """
    模型训练流程
    :param model: 模型
    :param train_loader: 训练数据集
    :param val_loader: 验证数据集
    :param optimizer: 优化器
    :param loss_function: 又称 criterion 损失函数
    :param num_epochs: 训练轮数
    :param device: 设备
    :param swanlab: 是否启用SwanLab日志记录
    :param slb_project: SwanLab项目名称
    :param slb_experiment: SwanLab实验名称
    :param slb_config: SwanLab配置字典
    """
    best_epoch = 0
    save_counter = 0
    early_stopping_counter = 0
    date_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    end_epoch = num_epochs
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    scheduler_name = scheduler.__class__.__name__ if scheduler else None
    loss_func_name = loss_function.__class__.__name__
    test_df = pd.read_csv(test_csv)

    
    swanlab_run = None
    if resume_tb_path:
        tb_dir = resume_tb_path
    else:
        tb_dir = os.path.join(results_dir, f'tensorBoard')
    ckpt_dir = os.path.join(results_dir, f'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)


    lr = optimizer.param_groups[0]["lr"] if not scheduler else scheduler.get_last_lr()[0]
    model_arch = str(summary(model, input_size=(1, 4, 128, 128, 128), device=DEVICE, verbose=False))
    # 初始化TensorBoard 
    if tb:
        writer = SummaryWriter(tb_dir)
        writer.add_graph(model, input_to_model=torch.rand(1, 4, 128, 128, 128).to(DEVICE))
        custom_logger(model_arch, logs_path)
        
    # 初始化SwanLab
    if slb:
        if swanlab is None:
            raise ImportError("SwanLab未安装，请使用 pip install swanlab 安装")
        slb_config = slb_config or {}
        # setup_logging(log_file='training.log')
        # slb_config.update({
        #     "model": model_name,
        #     "optimizer": optimizer_name,
        #     "scheduler": scheduler_name,
        #     "loss_function": loss_func_name,
        #     "initial_lr": lr,
        #     "batch_size": train_loader.batch_size,
        #     "num_epochs": num_epochs,
        #     "early_stopping": early_stopping_patience,
        # })
        swanlab_run = swanlab.init(
            project=slb_project,
            experiment_name=slb_experiment,
            config=slb_config,
            logdir='./logs'
        )
        swanlab_run.log({"model/summary": swanlab.Text(model_arch)})
        
    logger.info(
            f"\nmodel: {model_name}\n"\
            f"optimizer: {optimizer_name}\n"\
            f"scheduler: {scheduler_name}\n"\
            f"loss_function: {loss_func_name}\n"\
            f"initial_lr: {lr}\n"\
            f"batch_size: {train_loader.batch_size}\n"\
            f"num_epochs: {num_epochs}\n"\
            f"early_stopping: {early_stopping_patience}\n"\
    )
        
    # 初始化日志
    for epoch in range(start_epoch, end_epoch):
        epoch += 1
        interval = 1 if (epoch > 50) else interval
        """-------------------------------------- 训练过程 --------------------------------------------------"""
        logger.info(f"=== Training on [Epoch {epoch}/{end_epoch}] ===:")
        train_mean_loss = 0.0
        start_time = time.time()
        # 训练
        train_running_loss, train_et_loss, train_tc_loss, train_wt_loss = train_one_epoch(model, 
                                                                                          train_loader, 
                                                                                          scaler, 
                                                                                          optimizer, 
                                                                                          loss_function, 
                                                                                          device
                                                                                          )
        
        # 计算平均loss
        train_mean_loss =  train_running_loss / len(train_loader) 
        mean_train_et_loss = train_et_loss / len(train_loader)
        mean_train_tc_loss = train_tc_loss / len(train_loader)
        mean_train_wt_loss = train_wt_loss / len(train_loader)
        
        if scheduler_name == 'CosineAnnealingLR' and epoch > scheduler_start_epoch:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.warning(f"lr reduce to {current_lr}")
            
        # 记录训练指标
        if tb and writer:
            writer.add_scalars(f'{loss_func_name}/train',
                           {
                               'Mean':train_mean_loss, 
                               'ET': mean_train_et_loss, 
                               'TC': mean_train_tc_loss, 
                               'WT': mean_train_wt_loss
                               }, epoch)
        if swanlab_run:
            swanlab.log({
                "train/loss": train_mean_loss,
                "train/ET_loss": mean_train_et_loss,
                "train/TC_loss": mean_train_tc_loss,
                "train/WT_loss": mean_train_wt_loss,
                "learning_rate": current_lr,
            }, step=epoch)
            
        end_time = time.time()
        train_cost_time = end_time - start_time
        logger.info(f"- Train mean loss: {train_mean_loss:.4f}\n"
                    f"- ET loss: {mean_train_et_loss:.4f}\n"
                    f"- TC loss: {mean_train_tc_loss:.4f}\n"
                    f"- WT loss: {mean_train_wt_loss:.4f}\n"
                    f"- Cost time: {train_cost_time/60:.2f}mins ⏱️\n")
        
        """-------------------------------------- 验证过程 --------------------------------------------------"""
        if epoch % interval == 0:
            logger.info(f"=== Validating on [Epoch {epoch}/{end_epoch}] ===:")
            
            start_time = time.time()
            val_running_loss, val_et_loss, val_tc_loss, val_wt_loss, Metrics_list= val_one_epoch(model, 
                                                                                                 Metrics, 
                                                                                                 val_loader, 
                                                                                                 loss_function, 
                                                                                                 device,
                                                                                                 hd95=False
                                                                                                 )
            
            val_mean_loss = val_running_loss / len(val_loader)
            mean_val_et_loss = val_et_loss / len(val_loader)
            mean_val_tc_loss = val_tc_loss / len(val_loader)
            mean_val_wt_loss = val_wt_loss / len(val_loader)
            Metrics_list = Metrics_list / len(val_loader)
            
            val_scores = {}
            val_scores['epoch'] = epoch
            val_scores['Dice_scores'] = Metrics_list[0] 
            val_scores['Jaccard_scores'] = Metrics_list[1]
            val_scores['Precision_scores'] = Metrics_list[2]
            val_scores['Recall_scores'] = Metrics_list[3]
            
            """-------------------------------------- 记录验证指标 --------------------------------------------"""
            # TensorBoard记录
            if tb and writer:
                writer.add_scalars(f'{loss_func_name}/val', 
                               {
                                   'Mean':val_mean_loss, 
                                   'ET': mean_val_et_loss, 
                                   'TC': mean_val_tc_loss, 
                                   'WT': mean_val_wt_loss
                                }, 
                               epoch)
                writer.add_scalars(f'{loss_func_name}/Mean', 
                                {
                                    'Train':train_mean_loss,
                                    'Val':val_mean_loss
                                    }, 
                                epoch)
                writer.add_scalars(f'{loss_func_name}/ET',
                                {
                                    'Train':mean_train_et_loss, 
                                    'Val':mean_val_et_loss
                                    }, 
                                epoch)
                writer.add_scalars(f'{loss_func_name}/TC',
                                {
                                    'Train':mean_train_tc_loss, 
                                    'Val':mean_val_tc_loss
                                    }, 
                                epoch)
                writer.add_scalars(f'{loss_func_name}/WT',
                                {
                                    'Train':mean_train_wt_loss, 
                                    'Val':mean_val_wt_loss
                                    }, 
                                epoch)
                
                writer.add_scalars('metrics/Dice_coeff',
                                    {
                                        'Mean':val_scores['Dice_scores'][0],
                                        'ET': val_scores['Dice_scores'][1], 
                                        'TC': val_scores['Dice_scores'][2],
                                        'WT': val_scores['Dice_scores'][3]
                                        },
                                    epoch)
                # Jaccard指数
                writer.add_scalars('metrics/Jaccard_index', 
                                   {
                                       'Mean':val_scores['Jaccard_scores'][0],
                                       'ET': val_scores['Jaccard_scores'][1], 
                                       'TC': val_scores['Jaccard_scores'][2], 
                                       'WT': val_scores['Jaccard_scores'][3]
                                       },
                                    epoch)   
                # 召回率
                writer.add_scalars('metrics/Recall', 
                                    {
                                        'Mean':val_scores['Recall_scores'][0],
                                        'ET': val_scores['Recall_scores'][1],
                                        'TC': val_scores['Recall_scores'][2],
                                        'WT': val_scores['Recall_scores'][3]
                                        },
                                    epoch)    
                if epoch == start_epoch+1:
                    start_tensorboard(tb_dir, port=TB_PORT)
            # SwanLab记录
            if slb and swanlab_run:
                swanlab.log({
                    f"{loss_func_name}/Mean": {
                        "train": train_mean_loss,
                        "val": val_mean_loss
                    },
                    f"{loss_func_name}/ET":{
                        "train": mean_train_et_loss,
                        "val": mean_val_et_loss
                    },
                    f"{loss_func_name}/TC":{
                        "train": mean_train_tc_loss,  
                        "val": mean_val_tc_loss
                    },
                    f"{loss_func_name}/WT":{
                        "train": mean_train_wt_loss,
                        "val": mean_val_wt_loss 
                    },
                }, step=epoch)
                
                swanlab.log({
                    "metrics/Dice": val_scores['Dice_scores'][0],
                    "metrics/Dice_ET": val_scores['Dice_scores'][1],
                    "metrics/Dice_TC": val_scores['Dice_scores'][2],
                    "metrics/Dice_WT": val_scores['Dice_scores'][3],
                }, step=epoch)
                
                swanlab.log({
                    "metrics/Jaccard": val_scores['Jaccard_scores'][0],
                    "metrics/Jaccard_ET": val_scores['Jaccard_scores'][1],
                    "metrics/Jaccard_TC": val_scores['Jaccard_scores'][2],
                    "metrics/Jaccard_WT": val_scores['Jaccard_scores'][3],
                }, step=epoch)
                
                swanlab.log({
                    "metrics/Recall": val_scores['Recall_scores'][0],
                    "metrics/Recall_ET": val_scores['Recall_scores'][1],
                    "metrics/Recall_TC": val_scores['Recall_scores'][2],
                    "metrics/Recall_WT": val_scores['Recall_scores'][3],
                }, step=epoch)
                    
                
            end_time = time.time()
            val_cost_time = end_time - start_time
            
            """-------------------------------------- 打印指标 --------------------------------------------------"""
            metric_table_header = ["Metric_Name", "MEAN", "ET", "TC", "WT"]
            metric_table_left = ["Dice", "Jaccard", "Precision", "Recall"]

            val_info_str =  f"=== [Epoch {epoch}/{end_epoch}] ===\n"\
                            f"- Model:    {model_name}\n"\
                            f"- Optimizer:{optimizer_name}\n"\
                            f"- Scheduler:{scheduler_name}\n"\
                            f"- LossFunc: {loss_func_name}\n"\
                            f"- Lr:       {current_lr}\n"\
                            f"- val_cost_time:{val_cost_time:.4f}s ⏱️\n"\
                            f"- early_stopping: {early_stopping_patience}\n"

            
            metric_scores_mapping = {metric: val_scores[f"{metric}_scores"] for metric in metric_table_left}
            metric_table = [[metric,
                            format_value(metric_scores_mapping[metric][0]),
                            format_value(metric_scores_mapping[metric][1]),
                            format_value(metric_scores_mapping[metric][2]),
                            format_value(metric_scores_mapping[metric][3])] for metric in metric_table_left]
            loss_str = f"Mean Loss: {val_mean_loss:.4f}, ET: {mean_val_et_loss:.4f}, TC: {mean_val_tc_loss:.4f}, WT: {mean_val_wt_loss:.4f}\n"
            table_str = tabulate(metric_table, headers=metric_table_header, tablefmt='fancy_grid')
            metrics_info = val_info_str + table_str + '\n' + loss_str  
            custom_logger(metrics_info, logs_path)
            logger.info(metrics_info)
            
            """------------------------------------- 保存权重文件 --------------------------------------------"""
            best_dice = val_scores['Dice_scores'][0]
            if val_mean_loss < best_val_loss:
                early_stopping_counter = 0
                best_val_loss = val_mean_loss
                best_epoch = epoch
                with open(os.path.join(os.path.dirname(logs_path), "current_log.txt"), 'a') as f:
                    f.write(f"=== Best EPOCH {best_epoch} ===:\n"\
                            f"@ {get_current_time()}\n"\
                            f"current lr : {current_lr}\n"\
                            f"loss: Mean:{val_mean_loss:.4f}\t ET: {mean_val_et_loss:.4f}\t TC: {mean_val_tc_loss:.4f}\t WT: {mean_val_wt_loss:.4f}\n"
                            f"mean dice : {val_scores['Dice_scores'][0]:.4f}\t" \
                            f"ET : {val_scores['Dice_scores'][1]:.4f}\t"\
                            f"TC : {val_scores['Dice_scores'][2]:.4f}\t" \
                            f"WT : {val_scores['Dice_scores'][3]:.4f}\n\n")
                
                # 记录最佳指标到SwanLab
                if slb and swanlab_run:
                    swanlab.log({
                        "best/epoch": best_epoch,
                        "best/loss": best_val_loss,
                        "best/dice": best_dice,
                    }, step=epoch)
                
                
                # 删除旧的检查点文件
                def clean_old_ckpts(ckpt_dir, save_max):
                    # 获取所有检查点文件
                    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth') and os.path.isfile(os.path.join(ckpt_dir, f))] 

                    # 按修改时间排序
                    ckpt_files.sort(key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)))
                    
                    # 删除超限的旧文件
                    if len(ckpt_files) >= save_max:
                        os.remove(os.path.join(ckpt_dir, ckpt_files.pop(0)))
                        logger.critical(f"🗑️ Due to reach the max save amount, Removed {ckpt_files[0]}")
                        
                # 计算当前ckpt文件的个数
                ckpt_counter = sum(1 for f in os.listdir(ckpt_dir) 
                if f.lower().endswith('.pth') 
                and os.path.isfile(os.path.join(ckpt_dir, f)))
                
                # 保存save_max个最佳权重文件
                if ckpt_counter >= save_max:
                    clean_old_ckpts(ckpt_dir, save_max)
                    
                # 生成文件名时使用时间戳
                datetime_str = datetime.now().strftime('%Y%m%d%H%M%S')
                best_ckpt_path = os.path.join(ckpt_dir, 
                f"best_epoch{best_epoch}_loss{best_val_loss:.4f}_"
                f"dice{best_dice:.4f}_{datetime_str}.pth")
                
                save_checkpoint(
                    logger=logger,
                    model=model, 
                    optimizer=optimizer, 
                    scaler=scaler, 
                    epoch=best_epoch, 
                    best_val_loss=best_val_loss, 
                    checkpoint_path=best_ckpt_path, 
                    model_arch=model_arch,
                    scheduler=scheduler
                    )
            else:
                early_stopping_counter += 1
                logger.warning(f"😢😢😢Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                if early_stopping_counter >= early_stopping_patience:
                    logger.warning(f"🎃 Early stopping at epoch {epoch} due to no improvement in validation loss.")
                    break
                
    # 训练结束后关闭TensorBoard 和 SwanLab
    if tb and writer:
        writer.close() 
    if slb and swanlab_run:
        swanlab.finish()
        
    logger.critical(f"🥳🎉🎊Train finished. Best val loss: 👉{best_val_loss:.4f} at epoch {best_epoch}")
    # 保存最终模型
    logger.info(f"🛠️ 准备保存最终模型.......")
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)))
        latest_ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
        final_model_path = os.path.join(ckpt_dir, f'{model_name}_final_model.pth')
        shutil.copy(latest_ckpt_path, final_model_path)
        
        logger.info(f"✅ 最后一个权重文件已复制为 {final_model_path}")
    else:
        logger.error(f"⚠️ 没有找到任何权重文件在 {ckpt_dir}")
    # output_path = os.path.join(output_path, model_name, get_current_time())

    logger.critical("🥳🎉🎊 最终模型已保存")

    # 模型测试
    logger.info(f"🛠️ 准备测试模型.......")
    # 加载最优权重
    
    model, optimizer, scaler, _, _, _ = load_checkpoint(model, optimizer, scaler, final_model_path, load_weights_only=False)
    test_metrics_info = compute_finial_metric(model, Metrics, test_loader, loss_function, device, hd95=True)
    logger.critical(test_metrics_info)
    logger.info(f"🥳🎉🎊 模型测试已完成.......")
    custom_logger(test_metrics_info, logs_path)
    logger.info(f"测试集的指标已保存至{logs_path}")
    
    
def compute_finial_metric(model, Metrics, test_loader, loss_function, device, hd95=True):
    
    test_running_loss, test_et_loss, test_tc_loss, test_wt_loss, Metrics_list= val_one_epoch(model, 
                                                                                        Metrics, 
                                                                                        test_loader, 
                                                                                        loss_function, 
                                                                                        device,
                                                                                        hd95=hd95
                                                                                        )
    
    test_mean_loss = test_running_loss / len(test_loader)
    mean_test_et_loss = test_et_loss / len(test_loader)
    mean_test_tc_loss = test_tc_loss / len(test_loader)
    mean_test_wt_loss = test_wt_loss / len(test_loader)
    Metrics_list = Metrics_list / len(test_loader)
    
    test_scores = {}
    test_scores['Dice_scores'] = Metrics_list[0] 
    test_scores['Jaccard_scores'] = Metrics_list[1]
    test_scores['Precision_scores'] = Metrics_list[2]
    test_scores['Recall_scores'] = Metrics_list[3]
    test_scores['H95_scores'] = Metrics_list[4]
    
    metric_table_header = ["Metric_Name", "MEAN", "ET", "TC", "WT"]
    metric_table_left = ["Dice", "Jaccard", "Precision", "Recall", "H95"]

    test_info_str =  f"=== [FINAL TEST METRIC] ===\n"\
        
    metric_scores_mapping = {metric: test_scores[f"{metric}_scores"] for metric in metric_table_left}
    metric_table = [[metric,
                    format_value(metric_scores_mapping[metric][0]),
                    format_value(metric_scores_mapping[metric][1]),
                    format_value(metric_scores_mapping[metric][2]),
                    format_value(metric_scores_mapping[metric][3])] for metric in metric_table_left]
    
    loss_str = f"Mean Loss: {test_mean_loss:.4f};"\
               f"ET: {mean_test_et_loss:.4f};"\
               f"ET: {mean_test_et_loss:.4f};"\
               f"TC: {mean_test_tc_loss:.4f};" \
               f"WT: {mean_test_wt_loss:.4f}\n"
               
    table_str = tabulate(metric_table, headers=metric_table_header, tablefmt='fancy_grid')
    metrics_info = test_info_str + table_str + '\n' + loss_str  
    return metrics_info
    
        # val_scores['H95_scores'] = Metrics_list[4]
    
    
    
    
    # # 自动推理
    # attention_unet_scga_config = {
    #     'test_df': test_df,
    #     'test_loader': test_loader,
    #     'output_root': output_path,
    #     'model': model,
    #     'metricer': Metrics,
    #     'scaler': scaler,
    #     'optimizer': optimizer,
    #     'ckpt_path': final_model_path
    # }
    
    # inference(**attention_unet_scga_config)
    
    # print(f"🎉🎉🎉推理完成，结果保存在 {output_path}")
