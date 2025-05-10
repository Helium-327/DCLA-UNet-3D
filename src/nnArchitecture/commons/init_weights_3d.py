# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/03/12 22:00:59
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION:  权重初始化
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''
import torch.nn as nn
import torch.nn.functional as F


def init_weights_3d(m):
    """增强鲁棒性的3D权重初始化函数"""
    if isinstance(m, (nn.Conv3d)):
        nn.init.kaiming_normal_(
            m.weight, 
            mode='fan_in',
            nonlinearity='relu',
            a=0
        )
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(
            m.weight,
            mode='fan_out',
            nonlinearity='relu',
            a=0
        )
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        
    # 规范化层处理统一化
    elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
        if m.affine:  # 仅当具有可学习参数时进行初始化
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            # 对InstanceNorm增加额外初始化
            if isinstance(m, nn.InstanceNorm3d):
                # 在常数初始化基础上增加小扰动
                nn.init.normal_(m.weight, mean=1.0, std=0.01)
                
                