# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/03/12 22:03:07
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 上采样模块
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''

import torch.nn as nn
import torch.nn.functional as F

class UpSample(nn.Module):
    """3D Up Convolution"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super(UpSample, self).__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x):
        return self.up(x)