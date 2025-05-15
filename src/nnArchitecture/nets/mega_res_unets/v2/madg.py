# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 11:39:46
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v2.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiAxisDualAttnGate(nn.Module):
    """
    多轴双注意力门机制
    参数:
        F_g (int): 门控信号的通道数
        F_l (int): 跳跃连接的通道数
        F_inter (int): 中间特征维度，需能被4整除
    """
    def __init__(self, F_g, F_l, F_inter):
        
        assert F_inter % 4 == 0, "F_inter必须能被4整除"
        super().__init__()
        # 特征转换层
        # 上采样路径
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_inter, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=F_inter)  # 更好平衡通道独立性与相关性
        )
        
        # 跳跃连接路径（可考虑去掉，保留跳跃连接的特征）
        self.W_x = nn.Sequential(                       
            nn.Conv3d(F_l, F_inter, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=F_inter)
        )
        
        # 空间注意力 - 使用分组卷积提高效率
        self.spatial_att = CoordAttn3D(F_inter, F_l)  # 输出形状 [B,F_l,D,H,W]
        
        # 通道注意力 - 使用SE模块风格实现
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(F_inter, F_inter//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(F_inter//2, F_l, 1),
            nn.Sigmoid()
        )
        self.tau = nn.Parameter(torch.ones(1)*0.5)
        
        # 初始化权重
    def forward(self, g, x):
        # 特征融合 # [B, F_inter, D, H, W]
        fused = F.relu(self.W_g(g) + self.W_x(x)) 
        
        # 空间注意力 - 一次性计算三个维度 
        spatial_map = self.spatial_att(fused)  # [B, F_l, D, H, W]

        # 通道敏感的充标定
        # fused_mean = fused.mean(dim=(2, 3, 4), keepdim=True)
        channel_map = self.channel_att(fused)  # [B, F_l,1, 1, 1]
        
        tau = torch.sigmoid(self.tau)
        combined_weights = tau * spatial_map + (1 - tau) * channel_map
        return x * combined_weights
    

# @1: 添加归一化
# @2: 添加共享卷积层

class CoordAttn3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.base_conv = nn.Sequential(                     # @2
            nn.Conv3d(in_channels, in_channels//4, 1),  # 通道压缩
            nn.GroupNorm(4, in_channels//4),
            nn.ReLU(inplace=True)
        )
        self.d_conv = nn.Conv3d(in_channels//4, out_channels, (3, 1, 1), padding=(1, 0, 0))
        self.h_conv = nn.Conv3d(in_channels//4, out_channels, (1, 3, 1), padding=(0, 1, 0))
        self.w_conv = nn.Conv3d(in_channels//4, out_channels, (1, 1, 3), padding=(0, 0, 1))     
        
        self.dim_weights = nn.Parameter(torch.ones(3))   
        self.norm = nn.GroupNorm(4, out_channels)  #@1
        
    def forward(self, x):
        x = self.base_conv(x)
        d_feat = torch.sigmoid(self.norm(self.d_conv(x)))
        h_feat = torch.sigmoid(self.norm(self.h_conv(x)))
        w_feat = torch.sigmoid(self.norm(self.w_conv(x)))
        
        weights = F.softmax(self.dim_weights, dim=0)
        combined = weights[0] * d_feat + weights[1] * h_feat + weights[2] * w_feat
        return torch.sigmoid(self.norm(combined))