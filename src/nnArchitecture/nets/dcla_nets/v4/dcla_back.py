# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/04 11:16:59
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建DCLA UNet V2的DCLA模块
*      VERSION: v2.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''

import torch
import torch.nn as nn

class AdaptiveSpatialCondenser(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 kernel_size=[7,5,3], 
                 in_size=128, 
                 min_size=8,
                 fusion_mode='concat'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.in_size = in_size
        self.min_size = min_size
        self.fusion_mode = fusion_mode
        self.branches = self._build_multi_branchs()  # 动态构建下采样序列
        
    def _build_multi_branchs(self):
        return nn.ModuleList([self._build_single_branch(k) for k in self.kernel])
        
    def _build_single_branch(self, kernel_size):
        layers = []
        current_size = self.in_size
        
        # 动态构建下采样序列
        while current_size > self.min_size:
            layers.append(
                nn.Sequential(
                nn.Conv3d(
                    self.in_channels, 
                    self.out_channels, 
                    kernel_size=kernel_size, 
                    stride=2, 
                    padding=kernel_size//2
                    ),
                nn.BatchNorm3d(self.out_channels),
                nn.GELU()
            ))
            current_size = current_size // 2
        return nn.Sequential(*layers)
            
    def forward(self, x):
        branch_ouputs = [branch(x) for branch in self.branches]
        
        if self.fusion_mode == 'concat':
            return torch.cat(branch_ouputs, dim=1)
        elif self.fusion_mode == 'add':
            return torch.sum(torch.stack(branch_ouputs), dim=0)
        else:
            raise ValueError("Invalid fusion mode. Choose from 'concat' or 'add'.")

class DynamicCrossLevelAttention(nn.Module): #MSFA
    def __init__(self, 
                 ch_list, 
                 feats_size, 
                 min_size=8, 
                 squeeze_kernel=1,
                 down_kernel=[3,5,7], 
                 fusion_kernel=1,
                 fusion_mode='add'
                 ):
        """
        Args: 
            ch_list: 输入特征的通道数
            feats_size: 输入特征的空间尺寸
            min_size: 最小空间尺寸
            kernel_size: 卷积核大小, 可以是一个整数或一个元组 (k1, k2, k3, k4)
        """
        super().__init__()
        self.ch_list = ch_list
        self.feats_size = feats_size
        self.min_size = min_size
        self.kernel_size = down_kernel if isinstance(down_kernel, list) else [down_kernel]
        self.fusion_mode = fusion_mode
        self.squeeze_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        
        # if isinstance(self.kernel_size, int):
        for ch in self.ch_list:
            self.squeeze_layers.append(
                nn.Sequential(
                    nn.Conv3d(ch, 1, kernel_size=squeeze_kernel, padding=squeeze_kernel//2),
                    nn.BatchNorm3d(1),
                    nn.GELU()
                    ))
        for feat_size in feats_size:
            self.down_layers.append(
                AdaptiveSpatialCondenser(
                    in_channels=1, 
                    out_channels=1, 
                    kernel_size=self.kernel_size, 
                    in_size=feat_size, 
                    min_size=min_size,
                    fusion_mode=self.fusion_mode  # 'concat' or 'add'
                )
            )
        self.conv = nn.Sequential(
            nn.Conv3d(len(self.kernel_size),
                      1, 
                      kernel_size=1, 
                      padding=0
                      ),
            nn.BatchNorm3d(1),
            nn.GELU(),
            nn.Conv3d(1, 1, kernel_size=1, padding=0)
        )
        self.fusion = nn.Conv3d(len(self.ch_list), 1, kernel_size=fusion_kernel, padding=fusion_kernel//2)

    def forward(self, encoder_feats, x):
        squeezed_feats = []
        
        # 压缩通道数
        for i , squeeze_layer in enumerate(self.squeeze_layers):
            squeezed_feats.append(squeeze_layer(encoder_feats[i]))

        downs = []
        
        # 压缩空间维度
        for i, feat in enumerate(squeezed_feats):
            need_down = (feat.size(2) != self.min_size)
            if need_down:
                down_feat = self.down_layers[i](feat)
            else:
                down_feat = feat
            if self.fusion_mode == 'concat':
                down_feat = self.conv(down_feat)
            elif self.fusion_mode == 'add':
                down_feat = down_feat
            else:
                raise ValueError("Invalid fusion mode. Choose from 'concat' or 'add'.")
            
            downs.append(down_feat.squeeze(1))
        # 特征融合
        fused = self.fusion(torch.stack(downs, dim=1))
        attn = torch.sigmoid(fused)
        
        out = attn * x + x
        return out
    
# class DynamicCrossLevelAttention(nn.Module): #MSFA
#     def __init__(self, 
#                  ch_list, 
#                  feats_size, 
#                  min_size=8, 
#                  squeeze_kernel=1,
#                  down_kernel=[3,5,7], 
#                  fusion_kernel=1,
#                  fusion_mode='add'
#                  ):
#         """
#         Args: 
#             ch_list: 输入特征的通道数
#             feats_size: 输入特征的空间尺寸
#             min_size: 最小空间尺寸
#             kernel_size: 卷积核大小, 可以是一个整数或一个元组 (k1, k2, k3, k4)
#         """
#         super().__init__()
#         self.ch_list = ch_list
#         self.feats_size = feats_size
#         self.min_size = min_size
#         self.kernel_size = down_kernel if isinstance(down_kernel, list) else [down_kernel]
#         self.fusion_mode = fusion_mode
#         self.squeeze_layers = nn.ModuleList()
#         self.down_layers = nn.ModuleList()
        
#         # if isinstance(self.kernel_size, int):
#         for ch in self.ch_list:
#             self.squeeze_layers.append(
#                 nn.Sequential(
#                     nn.Conv3d(ch, 1, kernel_size=squeeze_kernel, padding=squeeze_kernel//2),
#                     nn.BatchNorm3d(1),
#                     nn.GELU()
#                     ))
#         for feat_size in feats_size:
#             self.down_layers.append(
#                 AdaptiveSpatialCondenser(
#                     in_channels=1, 
#                     out_channels=1, 
#                     kernel_size=self.kernel_size, 
#                     in_size=feat_size, 
#                     min_size=min_size,
#                     fusion_mode=self.fusion_mode  # 'concat' or 'add'
#                 )
#             )
#         self.conv = nn.Sequential(
#             nn.Conv3d(len(self.kernel_size),
#                       1, 
#                       kernel_size=1, 
#                       padding=0
#                       ),
#             nn.BatchNorm3d(1),
#             nn.GELU(),
#             nn.Conv3d(1, 1, kernel_size=1, padding=0)
#         )
#         self.fusion = nn.Conv3d(len(self.ch_list), 1, kernel_size=fusion_kernel, padding=fusion_kernel//2)

#     def forward(self, encoder_feats, x):
#         squeezed_feats = []
        
#         # 压缩通道数
#         for i , squeeze_layer in enumerate(self.squeeze_layers):
#             squeezed_feats.append(squeeze_layer(encoder_feats[i]))

#         downs = []
        
#         # 压缩空间维度
#         for i, feat in enumerate(squeezed_feats):
#             need_down = (feat.size(2) != self.min_size)
#             if need_down:
#                 down_feat = self.down_layers[i](feat)
#             else:
#                 down_feat = feat
#             if self.fusion_mode == 'concat':
#                 down_feat = self.conv(down_feat)
#             elif self.fusion_mode == 'add':
#                 down_feat = down_feat
#             else:
#                 raise ValueError("Invalid fusion mode. Choose from 'concat' or 'add'.")
            
#             downs.append(down_feat.squeeze(1))
#         # 特征融合
#         fused = self.fusion(torch.stack(downs, dim=1))
#         attn = torch.sigmoid(fused)
        
#         out = attn * x + x
#         return out

import math
class ECA3Dv2(nn.Module):
    """3D版本ECA模块，适配体积数据"""
    def __init__(self, in_channels, out_channels, gamma=2, b=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 3D全局池化 [D,H,W] -> [1,1,1]
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 动态计算1D卷积核大小 (保持原论文公式)
        kernel_size = int(abs((math.log2(in_channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.attn = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                nn.ReLU(),
                nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.Sigmoid()
        ) 

    def forward(self, encoder_feats, x):
        b, c, _, _, _ = x.shape
        feats = [self.avg_pool(feat).view(b, -1) + self.max_pool(feat).view(b, -1) for feat in encoder_feats]
        y = torch.cat(feats, dim=1)
        y = y.unsqueeze(1)                       # [B,1,C]
        attn = self.attn(y).view(b, -1, 1, 1, 1)  # 跨通道交互 [B,C,1,1,1]
        return x * attn              # 3D广播相乘
    
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(x)