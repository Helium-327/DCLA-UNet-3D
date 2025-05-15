# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 12:44:25
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建  Mega Res UNet v4
*      VERSION: v9.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''

import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from nnArchitecture.commons import (
    init_weights_3d,
    UpSample
)

from nnArchitecture.nets.mega_res_unets.v8 import (
    MegaResConv2,
    ResConv3D_S_BN,
    DynamicCrossLevelAttention,
    AttentionGate,
    MultiAxisDualAttnGatev1,
    DenseASPP3D
)

from utils.test_unet import test_unet


class MegaResUNetv9(nn.Module):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(MegaResUNetv9, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = MegaResConv2(in_channels, f_list[0], kernel_size=kernel_size)
        self.Conv2 = MegaResConv2(f_list[0], f_list[1], kernel_size=kernel_size)
        self.Conv3 = MegaResConv2(f_list[1], f_list[2], kernel_size=kernel_size) 
        self.Conv4 = MegaResConv2(f_list[2], f_list[3], kernel_size=kernel_size)

        self.bottleneck = DenseASPP3D(in_channels=f_list[3], out_channels=f_list[3], dilations=[1, 2, 3])
        
        self.Up5 = UpSample(f_list[3], f_list[3], trilinear)
        self.Att5 = AttentionGate(F_g=f_list[3], F_l=f_list[3], F_inter=f_list[3]//2)
        self.UpConv5 = MegaResConv2(f_list[3]*2, f_list[3]//2, kernel_size=3)
        
        self.Up4 = UpSample(f_list[2], f_list[2], trilinear)
        self.Att4 = AttentionGate(F_g=f_list[2], F_l=f_list[2], F_inter=f_list[2]//2)
        self.UpConv4 = MegaResConv2(f_list[2]*2, f_list[2]//2, kernel_size=3)
        
        self.Up3 = UpSample(f_list[1], f_list[1], trilinear)
        self.Att3 = AttentionGate(F_g=f_list[1], F_l=f_list[1], F_inter=f_list[1]//2)
        self.UpConv3 = MegaResConv2(f_list[1]*2, f_list[1]//2, kernel_size=3)
        
        self.Up2 = UpSample(f_list[0], f_list[0], trilinear)
        self.Att2 = AttentionGate(F_g=f_list[0], F_l=f_list[0], F_inter=f_list[0]//2)
        self.UpConv2 = MegaResConv2(f_list[0]*2, f_list[0], kernel_size=3)
        
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
        
        # Bottleneck
        # x5 = self.dcla([x1, x2, x3, x4], x5)
        x5 = self.bottleneck(x5)      # [B, 256, D/16, H/16, W/16]
        
        # Decoder with Attention
        d5 = self.Up5(x5)               # [B, 256, D/8, H/8, W/8]
        x4 = self.Att5(g=d5, x=x4)      # [B, 256, D/8, H/8, W/8]
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)    # [B, 128, D/8, H/8, W/8]
        
        d4 = self.Up4(d5)        # [B, 128, D/4, H/4, W/4]
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)    # [B, 64, D/4, H/4, W/4]
        
        d3 = self.Up3(d4)        # [B, 64, D/2, H/2, W/2]
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)    # [B, 32, D/2, H/2, W/2]
        
        d2 = self.Up2(d3)        # [B, 32, D, H, W]
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)    # [B, 32, D, H, W]
        
        out = self.outc(d2)  # [B, out_channels, D, H, W]
        return out
    
class MegaResUNetv9_LKDCLA(MegaResUNetv9):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(MegaResUNetv9_LKDCLA, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            kernel_size=kernel_size,
            trilinear=trilinear)
        
        self.dcla =DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, kernel_size=7)
        
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
        
        # Bottleneck
        x5 = self.dcla([x1, x2, x3, x4], x5)
        x5 = self.bottleneck(x5)      # [B, 256, D/16, H/16, W/16]
        
        # Decoder with Attention
        d5 = self.Up5(x5)               # [B, 256, D/8, H/8, W/8]
        x4 = self.Att5(g=d5, x=x4)      # [B, 256, D/8, H/8, W/8]
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)    # [B, 128, D/8, H/8, W/8]
        
        d4 = self.Up4(d5)        # [B, 128, D/4, H/4, W/4]
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)    # [B, 64, D/4, H/4, W/4]
        
        d3 = self.Up3(d4)        # [B, 64, D/2, H/2, W/2]
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)    # [B, 32, D/2, H/2, W/2]
        
        d2 = self.Up2(d3)        # [B, 32, D, H, W]
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)    # [B, 32, D, H, W]
        
        out = self.outc(d2)  # [B, out_channels, D, H, W]
        return out

    
    
if __name__ == "__main__":
    test_unet(model_class=MegaResUNetv9_LKDCLA, batch_size=1)   


