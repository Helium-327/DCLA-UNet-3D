# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/06 15:35:00
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 动态跨层空间注意力模块
*      VERSION: v1.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn


class DynamicCrossLayerSpatialAttention(nn.Module): #MSFA
    def __init__(self, 
                 ch_list, 
                 feats_size, 
                 min_size=8, 
                 squeeze_kernel=1,
                 down_kernel=[3,5,7], 
                 fusion_kernel=1,
                 mode = 'add'
                 ):
        """
        Args: 
            ch_list: 输入特征的通道数
            feats_size: 输入特征的空间尺寸
            min_size: 最小空间尺寸
            kernel_size: 卷积核大小, 可以是一个整数或一个元组 (k1, k2, k3, k4)
        """
        super().__init__()
        self.mode = mode
        self.ch_list = ch_list
        self.feats_size = feats_size
        self.min_size = min_size
        self.kernel_size = down_kernel if isinstance(down_kernel, list) else [down_kernel]
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
                    min_size=min_size
                )
            )
        self.in_channel = len(self.ch_list) if self.mode == 'concat' else 1
        self.attn = nn.Sequential(
            nn.Conv3d(self.in_channel, 1, kernel_size=fusion_kernel, padding=fusion_kernel//2),
            # nn.BatchNorm3d(1),
            # nn.GELU(),
            # nn.Conv3d(1, 1, kernel_size=fusion_kernel, padding=fusion_kernel//2),
            # nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

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
            downs.append(down_feat)
        # 特征融合
        if self.mode =='add':
            feats = sum(downs)
        elif self.mode == 'concat':
            feats = torch.cat(downs, dim=1)
        else:
            raise ValueError("Invalid fusion mode. Choose from 'concat' or 'add'.")
        attn = self.attn(feats)
        out = attn * x
        return out

class AdaptiveSpatialCondenser(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 kernel_size=[7,5,3], 
                 in_size=128, 
                 min_size=8
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.in_size = in_size
        self.min_size = min_size
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
        return torch.sum(torch.stack(branch_ouputs), dim=0)


