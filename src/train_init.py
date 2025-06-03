# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/28 15:12:56
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION:  加载模型
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from lossFunc import *
from metrics import *

import os
import sys

# 消融实验网络
from nnArchitecture.nets import *
from model_registry import model_register

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_name):
    for group in model_register.values():
        if model_name in group:
            return group[model_name].to(DEVICE)
    raise ValueError(f"模型不存在，可用模型: {[n for g in model_register.values() for n in g]}")

    # return model
def load_loss(loss_name):
    register_loss = {
        'diceloss':     DiceLoss(),
        'focalloss':    FocalLoss(),
        'celoss':       CELoss()
    }
    
    if loss_name in register_loss.keys():
        return register_loss[loss_name]
    else:
        raise ValueError(f"{loss_name} is not in {register_loss.keys()}")
def load_optimizer(model, lr, wd):
    """加载优化器"""
    optimizer = AdamW(model.parameters(), lr=float(lr), betas=(0.9, 0.99), weight_decay=float(wd))
    return optimizer
def load_scheduler(optimizer, cosine_T_max, cosine_eta_min):
    """加载调度器"""
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cosine_T_max), eta_min=float(cosine_eta_min))
    return scheduler

    