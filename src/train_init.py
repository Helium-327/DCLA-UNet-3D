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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_name):
    # 模型注册
    model_register = {
        # 对比网络
        'UNet3D':                   UNet3D(in_channels=4, out_channels=4),
        'AttUNet3D':                AttUNet3D(in_channels=4, out_channels=4),
        'UNETR':                    UNETR(
                                        in_channels=4,
                                        out_channels=4,
                                        img_size=(128, 128, 128),  # 输入图像尺寸
                                        feature_size=16,
                                        dropout_rate=0.2,
                                        norm_name='instance',
                                        spatial_dims=3
                                    ),
        'UNETR_PP':                 UNETR_PP(
                                        in_channels=4,
                                        out_channels=4,  # 假设分割为2类
                                        feature_size=16,
                                        hidden_size=256,
                                        num_heads=8,
                                        pos_embed="perceptron",
                                        norm_name="instance",
                                        dropout_rate=0.1,
                                        depths=[3, 3, 3, 3],
                                        dims=[32, 64, 128, 256],
                                        conv_op=nn.Conv3d,
                                        do_ds=False,
                                    ),
        'SegFormer3D':              SegFormer3D(in_channels=4, out_channels=4),
        # 'Mamba3d':                  Mamba3d(in_channels=4, out_channels=4),
        # 'MogaNet':                  MogaNet(in_channels=4, out_channels=4),
        
        #! 基线网络
        'DWResUNet':                            DWResUNet(in_channels=4, out_channels=4),
        'ResUNetBaseline_S':                    ResUNetBaseline_S(in_channels=4, out_channels=4),
        'ResUNetBaseline_M':                    ResUNetBaseline_M(in_channels=4, out_channels=4),

        #! DCLA_UNet v1
        'DCLA_UNet_v1':                         DCLA_UNet_v1(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_DCLA_v1':            ResUNetBaseline_S_DCLA_v1(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_SLK_v1':             ResUNetBaseline_S_SLK_v1(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_DCLA_SLK_v1':        ResUNetBaseline_S_DCLA_SLK_v1(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_MSF_v1':             ResUNetBaseline_S_MSF_v1(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_DCLA_MSF_v1':        ResUNetBaseline_S_DCLA_MSF_v1(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_SLK_MSF_v1':         ResUNetBaseline_S_SLK_MSF_v1(in_channels=4, out_channels=4),        

        #! DCLA_UNet v2
        'DCLA_UNet_v2':                         DCLA_UNet_v2(in_channels=4, out_channels=4),
        'DCLA_UNet_v2_1':                       DCLA_UNet_v2_1(in_channels=4, out_channels=4),
        'DCLA_UNet_v2_2':                       DCLA_UNet_v2_2(in_channels=4, out_channels=4),
        'DCLA_UNet_v2_3':                       DCLA_UNet_v2_3(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_DCLA_v2':            ResUNetBaseline_S_DCLA_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_DCLAv1_v2':          ResUNetBaseline_S_DCLAv1_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_SLK_v2':             ResUNetBaseline_S_SLK_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_SLKv1_v2':           ResUNetBaseline_S_SLKv1_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_SLKv2_v2':           ResUNetBaseline_S_SLKv2_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_DCLA_SLK_v2':        ResUNetBaseline_S_DCLA_SLK_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_DCLA_SLKv1_v2':      ResUNetBaseline_S_DCLA_SLKv1_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_DCLA_SLKv2_v2':      ResUNetBaseline_S_DCLA_SLKv2_v2(in_channels=4, out_channels=4),
        "ResUNetBaseline_S_DCLAv1_SLKv2_v2":    ResUNetBaseline_S_DCLAv1_SLKv2_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_MSF_v2':             ResUNetBaseline_S_MSF_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_DCLA_MSF_v2':        ResUNetBaseline_S_DCLA_MSF_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_DCLAv1_MSF_v2':      ResUNetBaseline_S_DCLAv1_MSF_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_SLK_MSF_v2':         ResUNetBaseline_S_SLK_MSF_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_SLKv1_MSF_v2':       ResUNetBaseline_S_SLKv1_MSF_v2(in_channels=4, out_channels=4),
        'ResUNetBaseline_S_SLKv2_MSF_v2':       ResUNetBaseline_S_SLKv2_MSF_v2(in_channels=4, out_channels=4),
    }
    # 加载模型
    
    if model_name in model_register.keys():
        return model_register[model_name].to(DEVICE)
    else:
        raise ValueError(f"{model_name} is not in {model_register.keys()}")

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

    