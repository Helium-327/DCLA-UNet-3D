# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 11:38:01
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建  Mega Res UNet 的 注意力门控 (Attn Gate)
*      VERSION: v5.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''

import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """轴向注意力门控模块"""
    def __init__(self, F_g, F_l, F_inter):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_inter, kernel_size=1),
            nn.InstanceNorm3d(F_inter)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_inter, kernel_size=1),
            nn.InstanceNorm3d(F_inter)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_inter, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        # g: 上采样后的特征图
        # x: 跳跃连接的特征图
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # out = self.axial_attns(F.relu(g1 + x1))
        out = F.relu(g1 + x1)
        psi = self.psi(out)
        out = x * psi                   # [B, 256, 16, 16, 16]
        return out

