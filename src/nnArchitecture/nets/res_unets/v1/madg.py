# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 15:56:25
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
import torch.nn.functional as F
# 多轴分离动态注意力门
class MultiAxisDualAttnGate(nn.Module):
# ✨ 创新点1： MADAG 模块 —— 空间维度分离的注意力模块
# ✨ 创新点2： 动态维度加权融合 - 动态调整平衡深度、高度和宽度维度的重要性
# ✨ 创新点3： 动态维度权重参数 - 允许模型在不同维度上学习不同的权重
# ✨ 创新点4： 自适应平均池化 - 用于通道注意力模块
    def __init__(self, F_g, F_l, F_inter):
        super().__init__()
        # 特征转换层
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_inter, kernel_size=1),
            nn.InstanceNorm3d(F_inter, affine=True)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_inter, kernel_size=1),
            nn.InstanceNorm3d(F_inter, affine=True)
        )
        
        # 空间注意力 - 使用分组卷积提高效率
        self.spatial_att = nn.Conv3d(F_inter, 3, kernel_size=3, padding=1, groups=1) 
        self.norm_d = nn.InstanceNorm3d(1, affine=True)             # 深度维度归一化
        self.norm_h = nn.InstanceNorm3d(1, affine=True)             # 高度维度归一化
        self.norm_w = nn.InstanceNorm3d(1, affine=True)             # 宽度维度归一化
        
        # 通道注意力 - 使用SE模块风格实现
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(F_inter, F_inter//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(F_inter//4, F_l, kernel_size=1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor([0.5]))  # 初始化平衡系数
        self.beta = nn.Parameter(torch.ones(3))         # 维度权重参数
        
        nn.init.constant_(self.alpha, 0.5)
        nn.init.constant_(self.beta, 1.0)
        
        # 初始化权重
    def forward(self, g, x):
        # 特征融合
        fused = F.relu(self.W_g(g) + self.W_x(x))
        
        # 空间注意力 - 一次性计算三个维度
        spatial_features = self.spatial_att(fused)  # [B, 3, D, H, W]
        
        # 三个维度的注意力
        feat_d = spatial_features[:, 0:1, :, :, :]      # 深度分量
        feat_h = spatial_features[:, 1:2, :, :, :]      # 高度分量
        feat_w = spatial_features[:, 2:3, :, :, :]      # 宽度分量
        
        
        # 各个维度归一化与激活
        attn_d = F.softmax(self.norm_d(feat_d), dim=2)  # 沿深度维度
        attn_h = F.softmax(self.norm_h(feat_h), dim=3)  # 沿高度维度
        attn_w = F.softmax(self.norm_w(feat_w), dim=4)  # 沿宽度维度
        
        # 动态维度加权融合
        beta = F.softmax(self.beta, dim=0)
        spatial_attn = beta[0] * attn_d + beta[1] * attn_h + beta[2] * attn_w
        
        # 通道注意力
        channel_weight = self.channel_att(fused)
        
        # 双注意力融合
        combined_attn = self.alpha * spatial_attn + (1-self.alpha) * channel_weight
        
        return x * combined_attn.expand_as(x)