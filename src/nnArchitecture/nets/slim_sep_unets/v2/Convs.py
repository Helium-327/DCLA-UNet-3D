# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 13:47:50
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建 Slim Sep UNet的 卷积模块
*      VERSION: v2.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn
import os
import sys

class SepConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups= None):
        super().__init__()  # 首先调用父类的 __init__ 方法
        
        k_eq = (kernel_size - 1)*dilation + 1
        p_eq = (k_eq - 1)//2
        assert k_eq % 2 == 1, "kernel_size must be odd"
        
        self.AxialConv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(kernel_size,1, 1),
                stride=stride,
                padding=(p_eq,0, 0),
                groups=groups if groups is not None else in_channels,
                dilation=dilation
            ),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1,kernel_size, 1),
                stride=stride,
                padding=(0,p_eq, 0),
                groups=groups if groups is not None else in_channels,
                dilation=dilation
            ),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1,1, kernel_size),
                stride=stride,
                padding=(0,0, p_eq),
                groups=groups if groups is not None else in_channels,
                dilation=dilation
            )
        )
        # 添加一个点卷积来改变通道数
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.AxialConv(x)
        x = self.pointwise(x)
        return x
    
class MultiKernelSlimUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=[1,2,3], groups=None):
        super().__init__()
        
        self.sep_branchs = nn.ModuleList()
        for d in dilations:
            self.sep_branchs.append(SepConv3d(
                in_channels,
                out_channels//2, 
                kernel_size=kernel_size, 
                stride=stride, 
                dilation=d,
                groups=groups if groups is not None else in_channels
                ))

        self.fusion = nn.Sequential(
            nn.Conv3d(
            in_channels=(out_channels//2)*len(dilations),
            out_channels=out_channels,
            kernel_size=1
            ),
            nn.BatchNorm3d(out_channels)
            )
        # self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.gelu = nn.GELU()
    
    def forward(self, x):
        sep_out = []
        for sep in self.sep_branchs:
            sep_out.append(sep(x))
        sep_out = torch.cat(sep_out, dim=1)
        out = self.fusion(sep_out)
        out += self.residual(x)
        return self.gelu(out)

class MultiKernelSlimUpBlockv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=[1,2,3], groups=None):
        """
        初始化 MultiKernelSlimUpBlockv2 模块。

        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int, optional): 卷积核大小. 默认为 3
            stride (int, optional): 卷积步长. 默认为 1
            dilations (list, optional): 空洞卷积的扩张率列表. 默认为 [1,2,3]
            groups (int, optional): 分组卷积的组数. 默认为 None
        """
        super().__init__()
        
        # 创建多个具有不同扩张率的分离卷积分支
        self.sep_branchs = nn.ModuleList()
        for d in dilations:
            self.sep_branchs.append(SepConv3d(
                in_channels,
                out_channels//2,  # 每个分支的输出通道数为总输出通道数的一半
                kernel_size=kernel_size, 
                stride=stride, 
                dilation=d,
                groups=groups if groups is not None else in_channels
                ))

        # 融合不同分支的特征
        self.fusion = nn.Sequential(
            nn.Conv3d(
            in_channels=out_channels//2,
            out_channels=out_channels,
            kernel_size=1
            ),
            nn.BatchNorm3d(out_channels)
            )
        
        # 残差连接，如果输入输出通道数不同，使用1x1卷积调整；否则使用恒等映射
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # GELU激活函数
        self.gelu = nn.GELU()