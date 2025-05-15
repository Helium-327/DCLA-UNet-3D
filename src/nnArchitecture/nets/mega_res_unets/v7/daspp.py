# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 12:33:40
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v7.0
*      FEATURES: 
                1. 轻量化改进
                2. 使用AxialDWConv
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
                    AxialDWConv(in_channels=current_channels, 
                              out_channels=growth_channels, 
                              kernel_size=3, 
                              dilation=d, 
                              ),
                    nn.BatchNorm3d(growth_channels),
                    nn.GELU()
                    # nn.Dropout3d(dropout)
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
    
class AxialDWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()  # 首先调用父类的 __init__ 方法
        
        k_eq = (kernel_size - 1)*dilation + 1
        p_eq = (k_eq - 1)//2
        assert k_eq % 2 == 1, "kernel_size must be odd"
        
        self.AxialConv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(kernel_size,1, 1),
                stride=stride,
                padding=(p_eq,0, 0),
                groups=in_channels,
                dilation=dilation
            ),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1,kernel_size, 1),
                stride=stride,
                padding=(0,p_eq, 0),
                groups=in_channels,
                dilation=dilation
            ),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1,1, kernel_size),
                stride=stride,
                padding=(0,0, p_eq),
                groups=in_channels,
                dilation=dilation
            )
        )
        
        # 添加一个点卷积来改变通道数
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.AxialConv(x)
        x = self.pointwise(x)
        return x