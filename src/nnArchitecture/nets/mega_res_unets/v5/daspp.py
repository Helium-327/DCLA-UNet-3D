# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 12:33:40
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v5.0
*      FEATURES: # @1：使用更加通用的形式，便于更换dilation
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn

## 改进点：
# @1：使用更加通用的形式，便于更换dilation
class DenseASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, reduce_rate=4, dilations=[1,2,3], dropout=0.2):
        super(DenseASPP3D, self).__init__()
        
        self.layers = nn.ModuleList()
        current_channels = in_channels
        growth_channels = out_channels//reduce_rate
        
        for d in dilations:
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(current_channels, growth_channels, 3, dilation=d, padding=d),
                    nn.GroupNorm(4, growth_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(dropout)
            ))
            current_channels += growth_channels
            
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, growth_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(growth_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        ) 
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels + 3*(growth_channels), out_channels, 1),
            nn.GroupNorm(8, out_channels),
        )
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            
            in_feat = torch.cat(features, dim=1)
            output = layer(in_feat)
            features.append(output)
    
        out = torch.cat(features, dim=1)
        out = self.fusion(out)
        
        channel_attented = self.global_avg(x)
        out = out * channel_attented
        
        out = self.act(out)
        return out