# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 11:28:15
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建 Mega Res UNet 卷积模块
*      VERSION: v5.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn

class MegaResConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dilation=1):
        super().__init__()
        # self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.depthwise =nn.Sequential(
            nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size,1, 1),
            stride=stride,
            padding=(kernel_size//2,0, 0),
            groups=in_channels
            ),
            nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1,kernel_size, 1),
            stride=stride,
            padding=(0,kernel_size//2, 0),
            groups=in_channels
            ),
            nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1,1, kernel_size),
            stride=stride,
            padding=(0,0, kernel_size//2),
            groups=in_channels
            ))
            
        self.pointwise = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1
        )
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.gelu = nn.GELU()
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.conv(out)
        out += self.residual(x)
        return self.gelu(out)
    
class ResConv3D_S_BN(nn.Module):
    """(conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=0.2, groups=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=1),
            nn.Conv3d(out_channels, 
                      out_channels, 
                      kernel_size=kernel_size, 
                      stride=stride,
                      padding=padding
                      ),
            nn.BatchNorm3d(out_channels),
        )
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(p=dropout_rate)  # 添加 Dropout 层
    def forward(self, x):
        out = self.double_conv(x) + self.residual(x)
        return self.drop(self.relu(out))