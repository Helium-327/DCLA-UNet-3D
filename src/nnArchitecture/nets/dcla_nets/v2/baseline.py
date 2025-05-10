# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 11:00:24
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建 DCLA UNet V2 的 Baseline
*      VERSION: v1.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from nnArchitecture.commons import (
    init_weights_3d, 
    UpSample,
    ResConv3D_S_BN
)
from utils.test_unet import test_unet

class ResUNetBaseline_S(nn.Module):
    # 3.08 M
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True, dropout_rate=0):
        super(ResUNetBaseline_S, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = ResConv3D_S_BN(in_channels, f_list[0], dropout_rate=dropout_rate)
        self.Conv2 = ResConv3D_S_BN(f_list[0], f_list[1], dropout_rate=dropout_rate)
        self.Conv3 = ResConv3D_S_BN(f_list[1], f_list[2], dropout_rate=dropout_rate)
        self.Conv4 = ResConv3D_S_BN(f_list[2], f_list[3], dropout_rate=dropout_rate)
        
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.UpConv4 = ResConv3D_S_BN(f_list[3]*2, f_list[3]//2, dropout_rate=dropout_rate)
        
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.UpConv3 = ResConv3D_S_BN(f_list[2]*2, f_list[2]//2, dropout_rate=dropout_rate)
        
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.UpConv2 = ResConv3D_S_BN(f_list[1]*2, f_list[1]//2, dropout_rate=dropout_rate)
        
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        self.UpConv1 = ResConv3D_S_BN(f_list[0]*2, f_list[0], dropout_rate=dropout_rate)
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
        self.apply(init_weights_3d)  # 初始化权重
        
    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)                # [B, 32, D, H, W]
        x2 = self.MaxPool(x1)
        x2 = self.Conv2(x2)      # [B, 64, D/2, H/2, W/2]
        x3 = self.MaxPool(x2)
        x3 = self.Conv3(x3)      # [B, 128, D/4, H/4, W/4]
        x4 = self.MaxPool(x3)
        x4 = self.Conv4(x4)      # [B, 256, D/8, H/8, W/8]
        x5 = self.MaxPool(x4)
        
        # Decoder with Attention
        d5 = self.Up4(x5)               # [B, 256, D/8, H/8, W/8]
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv4(d5)    # [B, 128, D/8, H/8, W/8]
        
        d4 = self.Up3(d5)        # [B, 128, D/4, H/4, W/4]
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv3(d4)    # [B, 64, D/4, H/4, W/4]
        
        d3 = self.Up2(d4)        # [B, 64, D/2, H/2, W/2]
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv2(d3)    # [B, 32, D/2, H/2, W/2]
        
        d2 = self.Up1(d3)        # [B, 32, D, H, W]
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv1(d2)    # [B, 32, D, H, W]
        
        out = self.outc(d2)  # [B, out_channels, D, H, W]
        return out
    

if __name__ == "__main__":
    test_unet(model_class=ResUNetBaseline_S, batch_size=1)  