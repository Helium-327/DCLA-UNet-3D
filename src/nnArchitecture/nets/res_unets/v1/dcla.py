# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 15:57:47
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

class AdaptiveSpatialCondenser(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=7, in_size=128, min_size=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.in_size = in_size
        self.min_size = min_size
        self.layers = self._build_layers()
        
        # 根据输入大小自动生成下采样层
    def _build_layers(self):
        layers = nn.ModuleList() 
        current_size = self.in_size
        while current_size > self.min_size:
            layers.append(
                nn.Sequential(
                nn.Conv3d(
                    self.in_channels, 
                    self.out_channels, 
                    kernel_size=self.kernel, 
                    stride=2, 
                    padding=self.kernel//2
                    ),
                nn.BatchNorm3d(self.out_channels),
                nn.ReLU(inplace=True)
            ))
            current_size = current_size // 2
        return layers
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DynamicCrossLevelAttention(nn.Module): #MSFA
    def __init__(self, ch_list, feats_size, min_size=8, kernel_size=3):
        super().__init__()
        self.ch_list = ch_list
        self.feats_size = feats_size
        self.min_size = min_size
        self.kernel_size = kernel_size
        
        self.squeeze_layers = nn.ModuleList()
        self.feat_size = feats_size
        for ch in ch_list:
            self.squeeze_layers.append(
                nn.Sequential(
                    nn.Conv3d(ch, 1, kernel_size=1),
                    nn.ReLU(inplace=True)
                    ))
        self.down_layers = nn.ModuleList()
        for feat_size in feats_size:
            self.down_layers.append(
                AdaptiveSpatialCondenser(
                    in_channels=1, 
                    out_channels=1, 
                    kernel_size=7, 
                    in_size=feat_size, 
                    min_size=8
                )
            )
        # self.upsample_layers = nn.Upsample(size=self.feat_size, mode='trilinear', align_corners=True)
        
        self.fusion = nn.Conv3d(len(ch_list), 1, kernel_size=1)

    def forward(self, encoder_feats, x):
        squeezed_feats = []
        
        # 压缩通道数
        for i , squeeze_layer in enumerate(self.squeeze_layers):
            squeezed_feats.append(squeeze_layer(encoder_feats[i]))

        downs = []
        
        # 压缩空间维度
        for i, feat in enumerate(squeezed_feats):
            if feat.shape[2:] != (self.min_size, self.min_size, self.min_size):
                downs.append(self.down_layers[i](feat).squeeze(1)) 
        # 特征融合
        fused = self.fusion(torch.stack(downs, dim=1))
        attn = torch.sigmoid(fused)
        
        out = attn * x
        return out