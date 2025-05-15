# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 13:49:03
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION:  构建 Slim Sep UNet的 下采样模块
*      VERSION: v5.0
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

class SlimDownBlock(nn.Module): # ResSepConv3D // LiteRes3D
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dilation=1, groups=None):
        super().__init__()
        self.depthwise =SepConv3d(
            in_channels,
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            groups=groups if groups is not None else in_channels
        )
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.gelu = nn.GELU()
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = self.conv(out)
        out += self.residual(x)
        return self.gelu(out)