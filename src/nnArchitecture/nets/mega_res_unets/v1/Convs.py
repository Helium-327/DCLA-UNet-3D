# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 11:28:15
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建 Mega Res UNet 卷积模块
*      VERSION: v1.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn

class MegaResConv(nn.Module): # ResDWConv3D
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.depthwise =nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            groups=out_channels
            )
        self.pointwise = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.depthwise(out)
        out = self.pointwise(out)
        out = self.bn(out)
        out += self.residual(x)
        return self.relu(out)