# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/06 15:53:45
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v1.0
=================================================
'''

import os
import time
import torch
import torch.nn as nn
import argparse
import logging
from train_swanlab import train
from tabulate import tabulate
from torch.utils.data import DataLoader
from metrics import EvaluationMetrics
from torch.amp import GradScaler
from train_init import load_model, load_loss, load_optimizer, load_scheduler
from utils import *
from lossFunc import *
from metrics import *
from datasets.BraTS2021 import DatasetSplitter


# 环境设置
# torch.backends.cudnn.benchmark = False         #! 加速固定输入/网络结构的训练，但需避免动态变化场景，如数据增强
# torch.backends.cudnn.deterministic = True     #! 确保结果可复现，但可能降低性能并引发兼容性问题

# 复现模式（测试/论文）
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# # 性能模式（默认推荐）
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

# !调试工具(不会用就不用，不然会后悔的🧐， 哥们)
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'    #! 检测每个操作的输入和输出张量的形状
# torch.autograd.set_detect_anomaly(True)     #! 检测梯度异常，但会降低性能（⚠️谨慎使用，哥们）

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
 # 混合精度训练
MetricsGo = EvaluationMetrics()  # 实例化评估指标类
    
# 定义颜色代码
class ColorCodes:
    GREY = "\033[0;37m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    RED = "\033[0;31m"
    BOLD_RED = "\033[1;31m"
    RESET = "\033[0m"

# 创建一个自定义的格式化器
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: ColorCodes.GREY + "%(asctime)s - %(levelname)s - %(message)s" + ColorCodes.RESET,
        logging.INFO: ColorCodes.GREEN + "%(asctime)s  - %(levelname)s - %(message)s" + ColorCodes.RESET,
        logging.WARNING: ColorCodes.YELLOW + "%(asctime)s  - %(levelname)s - %(message)s" + ColorCodes.RESET,
        logging.ERROR: ColorCodes.RED + "%(asctime)s  - %(levelname)s - %(message)s" + ColorCodes.RESET,
        logging.CRITICAL: ColorCodes.BOLD_RED + "%(asctime)s  - %(levelname)s - %(message)s" + ColorCodes.RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# 设置日志配置
def setup_colored_logging(file_name=None):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置最低级别，让所有日志都能被处理
    
    #  颜色化日志
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    
    file_handler = logging.FileHandler(filename=file_name, encoding='utf-8')
    file_handler.setFormatter(ColoredFormatter())
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 在你的主脚本开始时调用这个函数
logger = setup_colored_logging(file_name='Training.log')

def log_params(params, project_name, logs_path, logger, verbose=False):
    """记录训练参数"""
    params_dict = {'Parameter': [str(p[0]) for p in list(params.items())],
                   'Value': [str(p[1]) for p in list(params.items())]}
    params_header = ["Parameter", "Value"]
    custom_logger('='*40 + '\n' + "训练参数" +'\n' + '='*40 +'\n', logs_path, log_time=True)
    custom_logger(tabulate(params_dict, headers=params_header, tablefmt="grid"), logs_path)
    if verbose:
        logger.info(f'🧠 项目名：{project_name} \n' + \
            tabulate(params_dict, headers=params_header, tablefmt="fancy_grid"))
    

def load_data(args):
    from datasets import (
        Compose,
        RandomFlip3D,
        RandomCrop3D,
        CenterCrop3D,
        FrontGroundNormalize,
        RandomNoise3D,
        ToTensor
    )
    if args.datasets == 'BraTS2021':
        from datasets.BraTS2021 import BraTS21_3D as BraTS_Dataset
    elif args.datasets == 'BraTS2019':
        from datasets.BraTS2019 import BraTS19_3D as BraTS_Dataset
        
    """加载数据集"""
    TransMethods_train = Compose([
        RandomFlip3D(),
        RandomCrop3D((128, 128, 128)),
        FrontGroundNormalize(),
        RandomNoise3D(mean=0.0, std=(0, 0.1)),
        ToTensor()
    ])

    TransMethods_val = Compose([
        CenterCrop3D((128, 128, 128)),
        FrontGroundNormalize(),
        ToTensor(),
    ])
    
    TransMethods_test = Compose([
        CenterCrop3D((128, 128, 128)),
        FrontGroundNormalize(),
        ToTensor(),
    ])

    train_dataset = BraTS_Dataset(
        data_file=args.train_csv_path,
        transform=TransMethods_train,
        local_train=args.local,
        length=args.train_length,
    )
    
    val_dataset = BraTS_Dataset(
        data_file=args.val_csv_path,
        transform=TransMethods_val,
        local_train=args.local,
        length=args.val_length,
    )

    test_dataset = BraTS_Dataset(
        data_file=args.test_csv_path,
        transform=TransMethods_test,
        local_train=args.local,
        length=args.test_length,
    )
    
    setattr(args, 'train_length', len(train_dataset))
    logger.warning(f"训练集大小变成：{len(train_dataset)}")
    setattr(args, 'val_length', len(val_dataset))
    logger.warning(f"验证集大小变成：{len(val_dataset)}")
    setattr(args, 'test_length', len(test_dataset))
    logger.warning(f"测试大小变成：{len(test_dataset)}")
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True  # 减少 worker 初始化时间
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True  # 减少 worker 初始化时间
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True  # 减少 worker 初始化时间
    )
    return args, train_loader, val_loader, test_loader

def main(args):
    start_epoch = 0
    best_val_loss = float('inf')
    resume_tb_path = None
    scaler = GradScaler() # 混合精度训练
    
    """------------------------------------- 生成相对地址 --------------------------------------------"""
    train_csv_path = os.path.join(os.path.dirname(args.data_root), 'train.csv')
    val_csv_path = os.path.join(os.path.dirname(args.data_root), 'val.csv')
    test_csv_path = os.path.join(os.path.dirname(args.data_root), 'test.csv')
    
    setattr(args, 'train_csv_path', train_csv_path)
    setattr(args,'val_csv_path', val_csv_path)
    setattr(args,'test_csv_path', test_csv_path)
    
    """------------------------------------- 模型实例化、初始化 --------------------------------------------"""
    # 加载模型
    model = load_model(args.model_name)
    # 加载优化器
    optimizer = load_optimizer(model, args.lr, args.wd)

    # 加载调度器
    scheduler = load_scheduler(optimizer, args.cosine_T_max, args.cosine_eta_min)

    # 加载损失函数
    loss_function = load_loss(args.loss_type)
    total_params = sum(p.numel() for p in model.parameters())
    total_params = f'{total_params/1024**2:.2f} M'
    logger.info(f"Total number of parameters: {total_params}")
    setattr(args, 'total_parms', total_params)
    model_name = model.__class__.__name__
    """------------------------------------- 定义或获取路径 --------------------------------------------"""
    model_name_title = ('_').join([model_name, f'{get_current_date()}_lr{args.lr}_mlr{args.cosine_eta_min}_Tmax{args.cosine_T_max}_{args.epochs}_{args.early_stop_patience}'])
    if args.resume:
        resume_path = args.resume
        logger.info(f"Resuming training from {resume_path}")
        results_dir = os.path.join('/',*resume_path.split('/')[:-2])
        resume_tb_path = os.path.join(results_dir, 'tensorBoard')
        logs_dir = os.path.join(results_dir, 'logs')
        logs_file_name = [file for file in os.listdir(logs_dir) if file.endswith('.log')]
        logs_path = os.path.join(logs_dir, logs_file_name[0])
    else:
        os.makedirs(args.results_dir, exist_ok=True)
        results_dir = os.path.join(args.results_dir, model_name_title)       # TODO: 改成网络对应的文件夹
        os.makedirs(results_dir, exist_ok=True)
        logs_dir = os.path.join(results_dir, 'logs')
        logs_path = os.path.join(logs_dir, f'{get_current_date()}.log')
        os.makedirs(logs_dir, exist_ok=True)

    """------------------------------------- 记录当前实验内容 --------------------------------------------"""
    # exp_commit = args.commit if args.commit else input("请输入本次实验的更改内容: ")
    # write_commit_file(os.path.join(results_dir, 'commits.md'), exp_commit)


    """------------------------------------- 断点续传 --------------------------------------------"""
    if args.resume:
        logger.info(f"Resuming training from checkpoint {args.resume}")
        model, optimizer, scaler, start_epoch, best_val_loss, scheduler = load_checkpoint(model, 
                                                                                          optimizer, 
                                                                                          scaler, 
                                                                                          args.resume, 
                                                                                          scheduler
                                                                                          ) 
        logger.info(f"Loaded checkpoint {args.resume}")
        logger.info(f"Best val loss: {best_val_loss:.4f} ✈ epoch {start_epoch}")
        cutoff_tb_data(resume_tb_path, start_epoch)
        logger.warning(f"Refix resume tb data step {resume_tb_path} up to step {start_epoch}")

    # 分割数据集/判断是否重叠
    if args.data_split and args.datasets == 'BraTS2021':
        dataset_splitter = DatasetSplitter(args.data_root)
        dataset_splitter.collect_files()
        dataset_splitter.split(random_seed=RANDOM_SEED)
        dataset_splitter.save_csv(os.path.dirname(args.data_root))
    else:
        dataset_splitter = DatasetSplitter(args.data_root)
        dataset_splitter.diff_datasets(args.train_csv_path, args.val_csv_path, args.test_csv_path, logger)
        
    # 加载数据集
    args, train_loader, val_loader, test_loader = load_data(args)
    
    
    # 记录训练配置
    log_params(vars(args), args.slb_project, logs_path, logger, verbose=True)
    """------------------------------------- 训练模型 --------------------------------------------"""
    train(model=model, 
          Metrics=MetricsGo, 
          train_loader=train_loader,
          val_loader=val_loader, 
          test_loader=test_loader,
          scaler=scaler, 
          optimizer=optimizer,
          scheduler=scheduler,
          loss_function=loss_function,
          num_epochs=args.epochs, 
          device=DEVICE, 
          results_dir=results_dir,
          logs_path=logs_path,
          output_path=args.outputs_dir,
          start_epoch=start_epoch,
          best_val_loss=best_val_loss,
          test_csv=args.test_csv_path,
          interval=args.interval,
          save_max=args.save_max,
          early_stopping_patience=args.early_stop_patience,
          # tb 相关参数          
          tb=args.tb,
          # resume_tb_path=resume_tb_path
          
          # SwanLab相关参数pip
          slb=args.slb,
          slb_project=args.slb_project,
          slb_experiment=model_name_title,
          logger=logger,
          slb_config=args,
          )

if __name__ == '__main__':
    start_time = time.time()
    
    # 获取工作路径
    ROOT = "/root/workspace/DCLA-UNet"                         #TODO: 切换机器需要更改
    if not os.path.exists(ROOT):
        ROOT = "/root/autodl-tmp/DCLA-UNet/"
    
    initial_parser = argparse.ArgumentParser(add_help=False)
    
    #*** 固定参数 ***#
    initial_parser.add_argument('--data_root', 
                                type=str, 
                                default=os.path.join(ROOT, "data/BraTS2019/raw"), 
                                help='Path to the DATASET ROOT'
                                )
    initial_parser.add_argument('--outputs_dir', 
                                type=str, 
                                default=os.path.join(ROOT, 'outputs'),
                                help='Path to the DATASET ROOT'
                                )
    initial_parser.add_argument('--results_dir', 
                                type=str, 
                                default=os.path.join(ROOT, 'results'),
                                help='Path to the DATASET ROOT'
                                )
    initial_parser.add_argument('--in_channel', 
                                type=int, 
                                default=4,
                                help='Name of input channel'
                                )
    initial_parser.add_argument('--out_channel', 
                                type=int, 
                                default=4,
                                help='Name of input channel / num of classes'
                                )
    #*** 必传参数 ***#
    initial_parser.add_argument('--config', 
                                type=str, 
                                default=None, 
                                help='Path to the configuration YAML file'
                                )
    initial_parser.add_argument('--model_name', 
                                type=str, 
                                default="UNet3D",
                                help='Name of Model'
                                )
    initial_parser.add_argument('--slb_project', 
                                type=str, 
                                default='debug',
                                help='Name of SwanLab project'
                                )
    #*** 可选参数 ***#
    initial_parser.add_argument('--resume', 
                                type=str, 
                                default=False,
                                help='Path to the Resume checkpoint'
                                )
    initial_parser.add_argument('--loss_type', 
                                type=str, 
                                default="diceloss",
                                help='Type of Loss Function'
                                )
    initial_parser.add_argument('--optimizer_type', 
                                type=str, 
                                default="adamw",
                                help='Type of optimizer'
                                )
    initial_parser.add_argument('--epochs', 
                                type=int, 
                                default=100,
                                help='Num of epochs'
                                )
    initial_parser.add_argument('--batch_size', 
                                type=int, 
                                default=1,
                                help='Num of batch size'
                                )
    initial_parser.add_argument('--num_workers', 
                                type=int, 
                                default=8,
                                help='Num of workers'
                                )
    initial_parser.add_argument('--lr', 
                                type=float, 
                                default=3e-4,
                                help='Learning rate (required), recommend 1e-5'
                                )
    initial_parser.add_argument('--wd', 
                                type=float, 
                                default=1e-5,
                                help='Weight decay, recommend 1e-5'
                                )
    initial_parser.add_argument('--cosine_eta_min', 
                                type=float, 
                                default=1e-6,
                                help='min lr in cosine scheduler, recommend 1e-6'
                                )
    initial_parser.add_argument('--cosine_T_max', 
                                type=int, 
                                default=50,
                                help='max T in cosine scheduler, recommend the number which half of the epochs)'
                                )
    initial_parser.add_argument('--early_stop_patience', 
                                type=int, 
                                default=60,
                                help='Early Stop after epochs, recommend 60, to cover the half of epochs'
                                )
    initial_parser.add_argument('--scheduler_type', 
                                type=str, 
                                default="cosine",
                                help='Type of scheduler'
                                )
    initial_parser.add_argument('--save_max', 
                                type=int, 
                                default=2,
                                help='Max saved number'
                                )
    initial_parser.add_argument('--interval', 
                                type=int, 
                                default=1,
                                help='Early Stop after epochs'
                                )
    initial_parser.add_argument('--commit', 
                                type=str, 
                                default="Training",
                                help='commit'
                                )
    initial_parser.add_argument('--datasets', 
                                type=str, 
                                default="BraTS2019",
                                help='datasets name, BraTS2019 or BraTS2021'
                                )
    
    #*** 测试参数 ***#
    initial_parser.add_argument('--data_split', 
                                action='store_true',
                                help='Flag of splitting dataset'
                                )
    initial_parser.add_argument('--local', 
                                action='store_true',
                                help='Flag of partial dataset'
                                )
    initial_parser.add_argument('--train_length', 
                                type=int, 
                                default=70,
                                help='Num of partial train dataset'
                                )
    initial_parser.add_argument('--val_length', 
                                type=int, 
                                default=10,
                                help='Num of partial val dataset'
                                )
    initial_parser.add_argument('--test_length', 
                                type=int, 
                                default=10,
                                help='Num of partial test dataset'
                                )
    initial_parser.add_argument('--slb', 
                                action='store_true',
                                help='Flag of SwanLab'
                                )
    initial_parser.add_argument('--tb', 
                                action='store_true',
                                help='Flag of TensorBoard'
                                )
    initial_args = initial_parser.parse_args()
    
    end_time = time.time()
    logger.info(f"加载配置文件耗时: {end_time - start_time:.2f} s")
    
    # 传递合并后的参数
    main(args=initial_args)