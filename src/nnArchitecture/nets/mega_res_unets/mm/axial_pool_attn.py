# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 13:03:52
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 轴向池化注意力模块（显存优化版）
*      VERSION: v1.0
*      FEATURES:     1. 使用轴向池化降低计算复杂度
                    2. 并行分支结构保持各维度信息
                    3. 轻量级注意力机制
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn

# class AxialPoolAttn3D(nn.Module):
#     """
#     轴向池化注意力模块（显存优化版）
#     特点：
#     1. 使用轴向池化降低计算复杂度
#     2. 并行分支结构保持各维度信息
#     3. 轻量级注意力机制
#     """
    
#     def __init__(self, in_channels, out_channels):
#         super().__init__()  
        
#         self.d_pool = nn.Sequential(
#             nn.AdaptiveAvgPool3d((None, 1, 1)),  # 保持D维度
#             nn.Conv3d(in_channels, in_channels//2, 1),
#             nn.ReLU(inplace=True)
#         )
        
#         self.w_pool = nn.Sequential(
#             nn.AdaptiveAvgPool3d((1, 1, None)),  # 保持W维度
#             nn.Conv3d(in_channels, in_channels//2, 1),
#             nn.ReLU(inplace=True)
#         )
        
#         self.h_pool = nn.Sequential(
#             nn.AdaptiveAvgPool3d((1, None, 1)),  # 保持H维度
#             nn.Conv3d(in_channels, in_channels//2, 1),
#             nn.ReLU(inplace=True)
#         )
        
#         self.attn_gen = nn.Sequential(
#             nn.Conv3d(in_channels//2 * 3, out_channels, 1),  # 合并三个分支
#             nn.GroupNorm(4, out_channels),
#             nn.Sigmoid()
#         )
        
#         self.res_scale = nn.Parameter(torch.zeros(1))
#         self.res_conv = nn.Conv3d(in_channels, out_channels, 1)
        
#     def forward(self, x):
#         residual = self.res_conv(x)
#         # x = self.channel_compress(x)
        
#         # 各轴向池化
#         d_feat = self.d_pool(x)  # [B, C//16, D, 1, 1]
#         h_feat = self.h_pool(x)  # [B, C//16, 1, H, 1]
#         w_feat = self.w_pool(x)  # [B, C//16, 1, 1, W]
        
#         # 特征融合与注意力生成
#         combined = torch.cat([
#             d_feat.expand(-1,-1,-1,x.size(3),x.size(4)),
#             h_feat.expand(-1,-1,x.size(2),-1,x.size(4)),
#             w_feat.expand(-1,-1,x.size(2),x.size(3),-1)
#         ], dim=1)
        
#         attn_map = self.attn_gen(combined)
        
#         # 残差连接
#         return residual * (attn_map + self.res_scale)

class AxialPoolAttn3D(nn.Module):
    """
    轴向池化注意力模块（显存优化版）
    特点：
    1. 使用轴向池化降低计算复杂度
    2. 并行分支结构保持各维度信息
    3. 轻量级注意力机制
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.d_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),  # 压缩D维度
            nn.Conv3d(in_channels, in_channels//2, 1),
            nn.ReLU(inplace=True)
        )
        
        self.h_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, None)),  # 压缩H维度
            nn.Conv3d(in_channels, in_channels//2, 1),
            nn.ReLU(inplace=True)
        )
        
        self.w_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, None, 1)),  # 压缩W维度
            nn.Conv3d(in_channels, in_channels//2, 1),
            nn.ReLU(inplace=True)
        )
        
        self.attn_gen = nn.Sequential(
            nn.Conv3d(in_channels//2 * 3, out_channels, 1),  # 合并三个分支
            nn.GroupNorm(4, out_channels),
            nn.Sigmoid()
        )
        
        # self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.res_scale = nn.Parameter(torch.zeros(1))
        self.res_conv = nn.Conv3d(in_channels, out_channels, 1)
        
    def forward(self, x):
        residual = self.res_conv(x)
        # x = self.channel_compress(x)
        
        # 各轴向池化
        d_feat = self.d_pool(x)  # [B, C//16, D, 1, 1]
        h_feat = self.h_pool(x)  # [B, C//16, 1, H, 1]
        w_feat = self.w_pool(x)  # [B, C//16, 1, 1, W]
        
        # 特征融合与注意力生成
        combined = torch.cat([
            d_feat.expand(-1,-1,x.size(2),-1,-1),
            h_feat.expand(-1,-1,-1,x.size(3),-1),
            w_feat.expand(-1,-1,-1, -1,x.size(4))
        ], dim=1)
        
        attn_map = self.attn_gen(combined)
        
        # 残差连接
        return residual * (attn_map + self.res_scale)