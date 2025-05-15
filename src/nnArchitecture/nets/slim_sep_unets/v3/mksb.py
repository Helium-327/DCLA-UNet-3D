# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 13:50:35
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建 Slim Sep UNet的 上采样模块
*      VERSION: v3.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from nnArchitecture.nets.slim_sep_unets.v1.Convs import (
    SepConv3d
)

class MultiKernelSlimUpBlockv3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 5, 7], stride=1, groups=None):
        """
        初始化 MultiKernelSlimUpBlockv3 模块。
        改进点： 不使用dilation

        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int, optional): 卷积核大小. 默认为 3
            stride (int, optional): 卷积步长. 默认为 1
            dilations (list, optional): 空洞卷积的扩张率列表. 默认为 [1,2,3]
            groups (int, optional): 分组卷积的组数. 默认为 None
        """
        super().__init__()
        
        # 创建多个具有不同核的分离卷积分支
        self.sep_branchs = nn.ModuleList()
        for k in kernel_size:
            self.sep_branchs.append(SepConv3d(
                in_channels,
                out_channels*2,  # 每个分支的输出通道数为总输出通道数的一半
                kernel_size=k, 
                stride=stride, 
                groups=groups if groups is not None else in_channels
                ))

        # 融合不同分支的特征
        self.fusion = nn.Sequential(
            nn.Conv3d(
            in_channels=out_channels*2,
            out_channels=out_channels,
            kernel_size=1
            ),
            nn.BatchNorm3d(out_channels)
            )
        
        # 残差连接，如果输入输出通道数不同，使用1x1卷积调整；否则使用恒等映射
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # GELU激活函数
        self.gelu = nn.GELU()
    
    def forward(self, x):
        sep_out = 0
        for sep in self.sep_branchs:
            sep_out = torch.add(sep_out, sep(x))  # 使用加法而不是拼接
        out = self.fusion(sep_out)
        out += self.residual(x)
        return self.gelu(out)