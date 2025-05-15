# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 11:35:45
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION:  构建 Mega Res UNet 跨层注意力模块(DCLA)
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
        self.kernel_size = kernel_size
        self.squeeze_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        
        if isinstance(self.kernel_size, int):
            for ch in self.ch_list:
                self.squeeze_layers.append(
                    nn.Sequential(
                        nn.Conv3d(ch, 1, kernel_size=1),
                        nn.ReLU(inplace=True)
                        ))
            for feat_size in feats_size:
                self.down_layers.append(
                    AdaptiveSpatialCondenser(
                        in_channels=1, 
                        out_channels=1, 
                        kernel_size=self.kernel_size, 
                        in_size=feat_size, 
                        min_size=8
                    )
                )
        elif len(self.kernel_size) == 4:
            for k, ch in zip(self.kernel_size, self.ch_list):
                self.squeeze_layers.append(
                    nn.Sequential(
                        nn.Conv3d(ch, 1, kernel_size=k, padding=k//2, bias=False),
                        nn.ReLU(inplace=True)
                        ))
            for k, feat_size in zip(self.kernel_size, self.feats_size):
                self.down_layers.append(
                    AdaptiveSpatialCondenser(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=k,
                        in_size=feat_size,
                        min_size=8
                    )
                )
        else:
            raise ValueError("kernel_size must be an integer or a tuple of length 4.")
        # self.upsample_layers = nn.Upsample(size=self.feat_size, mode='trilinear', align_corners=True)
        
        self.fusion = nn.Conv3d(len(ch_list), 1, kernel_size=kernel_size, padding=kernel_size//2)

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
                down_feat = self.down_layers[i](feat).squeeze(1)
            else:
                down_feat = feat.squeeze(1)
            downs.append(down_feat)
        # 特征融合
        fused = self.fusion(torch.stack(downs, dim=1))
        attn = torch.sigmoid(fused)
        
        out = attn * x
        return out