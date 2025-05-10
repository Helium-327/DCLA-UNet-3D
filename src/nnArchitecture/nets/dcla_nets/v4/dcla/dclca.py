# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/06 15:40:11
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 动态跨层通道注意力模块
*      VERSION: v1.0
*      FEATURES: 
               1. 全局最大池化 + 全局平均池化
               2. ECA模块
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import math
import torch
import torch.nn as nn

class DynamicCrossLayerChannelsAttention(nn.Module):
    def __init__(self, in_channels, out_channels, gamma=2, b=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 3D全局池化 [D,H,W] -> [1,1,1]
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 动态计算1D卷积核大小 (保持原论文公式)
        kernel_size = int(abs((math.log2(in_channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.eca = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                # Swish(),
                # nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                nn.Sigmoid()
        ) 
        
    def forward(self, encoder_feats, x):
        batch_size, _, _, _, _ = x.shape
        feats = [
            self.avg_pool(feat).view(batch_size, -1) *
            self.max_pool(feat).view(batch_size, -1) 
            for feat in encoder_feats
        ]
        y = torch.cat(feats, dim=1)
        y = y.unsqueeze(1)                       # [B,1,C]
        attn = self.eca(y).view(batch_size, -1, 1, 1, 1)  # 跨通道交互 [B,C,1,1,1]
        return x * attn  
    
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(x)
