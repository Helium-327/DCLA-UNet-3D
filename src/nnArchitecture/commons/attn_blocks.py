# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/04/30 17:47:30
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION:  复现各种即插即用的注意力模块
*      VERSION: v1.0
*      FEATURES: 
               - CA Attention 3D
               - SA Attention 3D
               - CBAM Attention 3D
               - Self-Attention 3D
               - Axial Attention 3D
               
*      STATE:     TODO: 完善注意力模块
*      CHANGE ON: 
=================================================
'''

from SimpleITK import Sigmoid
import torch
import torch.nn as nn
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from nnArchitecture.commons import Swish

class ChannelAttention3D(nn.Module):
    pass


class SpatialAttention3D(nn.Module):
    pass


class CBAMAttention3D(nn.Module):
    pass


class SelfAttention3D(nn.Module):
    pass


class AxialAttention3D(nn.Module):
    pass


class LightweightSpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=3, mode='mul'):
        super(LightweightSpatialAttention3D, self).__init__()
        self.mode = mode
        self.in_channels = 2 if mode == 'concat' else 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_channels, 1, kernel_size, padding=kernel_size//2),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.Conv3d(1, 1, kernel_size, padding=kernel_size//2),
            nn.BatchNorm3d(1),
            nn.Sigmoid()      
        )
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        if self.mode == 'add':
            out = avg_out + max_out
        elif self.mode == 'concat':
            out = torch.cat([avg_out, max_out], dim=1)
        elif self.mode == 'mul':
            out = avg_out * max_out
        else:
            raise ValueError('mode error')
        attn = self.conv1(out)
        return attn

class LightweightChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(LightweightChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class LightweightChannelAttention3Dv2(nn.Module):
    def __init__(self, kernel_size=3):
        super(LightweightChannelAttention3Dv2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.eca = nn.Sequential(
            nn.Conv1d(1, 1 , kernel_size=kernel_size, padding=kernel_size//2),
            # Swish(),
            # nn.Conv1d(1, 1 , kernel_size=kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, _, _, _, _ = x.size()
        avg_out = (self.avg_pool(x) + self.max_pool(x)).view(batch_size, -1)
        attn = self.eca(avg_out.unsqueeze(1)).view(batch_size, -1, 1, 1, 1)
        return attn
    
    