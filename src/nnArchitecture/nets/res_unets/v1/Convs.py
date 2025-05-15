# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 15:37:29
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v1.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn

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
    
class ResConv3D_M_BN(nn.Module):
    """(conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, groups, kernel_size=3, stride=1, padding=1, dropout_rate=0.2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, 
                      out_channels, 
                      kernel_size=kernel_size, 
                      stride=stride,
                      padding=padding
                      ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
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
