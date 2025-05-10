# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/04 10:47:27
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建DCLA UNet V1的SLK模块
*      VERSION: v1.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn

from nnArchitecture.commons import (
    DepthwiseAxialConv3d
)

class SlimLargeKernelBlock(nn.Module): 
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3 
                ):
        super().__init__()
        self.depthwise =nn.Sequential(
                DepthwiseAxialConv3d(
                    in_channels,
                    in_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size
                ),
                nn.BatchNorm3d(in_channels),
                nn.GELU(),
                nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=1
                    ),
                nn.BatchNorm3d(out_channels),
        )
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.GELU()
        
    def forward(self, x):
        out = self.depthwise(x)
        
        out += self.residual(x)
        return self.act(out)