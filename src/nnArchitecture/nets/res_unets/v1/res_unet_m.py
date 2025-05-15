# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/03/08 14:49:30
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: ResUNet3D 
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from nnArchitecture.commons import (
    init_weights_3d,
    UpSample
)

from nnArchitecture.nets.res_unets.v1 import (
    ResConv3D_M_BN,
    AttentionGate,
    DenseASPP3D,
    MultiAxisDualAttnGate,
    DynamicCrossLevelAttention
)


from utils import test_unet

class ResUNet3D_M(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, groups=16, f_list=[32, 64, 128, 256], trilinear=True):
        super(ResUNet3D_M, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = ResConv3D_M_BN(in_channels, f_list[0], groups=1)
        self.Conv2 = ResConv3D_M_BN(f_list[0], f_list[1], groups=groups)
        self.Conv3 = ResConv3D_M_BN(f_list[1], f_list[2], groups=groups)
        self.Conv4 = ResConv3D_M_BN(f_list[2], f_list[3], groups=groups)
        
        self.bottleneck = ResConv3D_M_BN(f_list[3], f_list[3], groups=groups)
        
        self.Up5 = UpSample(f_list[3], f_list[3], trilinear)
        self.UpConv5 = ResConv3D_M_BN(f_list[3]*2, f_list[3]//2, groups=groups)
        
        self.Up4 = UpSample(f_list[2], f_list[2], trilinear)
        self.UpConv4 = ResConv3D_M_BN(f_list[2]*2, f_list[2]//2, groups=groups)
        
        self.Up3 = UpSample(f_list[1], f_list[1], trilinear)
        self.UpConv3 = ResConv3D_M_BN(f_list[1]*2, f_list[1]//2, groups=groups)
        
        self.Up2 = UpSample(f_list[0], f_list[0], trilinear)
        self.UpConv2 = ResConv3D_M_BN(f_list[0]*2, f_list[0], groups=groups)
        
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
        x5 = self.bottleneck(x5)      # [B, 256, D/16, H/16, W/16]
        
        # Decoder with Attention
        d5 = self.Up5(x5)               # [B, 256, D/8, H/8, W/8]
        # x4 = self.Att5(g=d5, x=x4)      # [B, 256, D/8, H/8, W/8]
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)    # [B, 128, D/8, H/8, W/8]
        
        d4 = self.Up4(d5)        # [B, 128, D/4, H/4, W/4]
        # x3 = self.Att4(g=d4, x=x3)s
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)    # [B, 64, D/4, H/4, W/4]
        
        d3 = self.Up3(d4)        # [B, 64, D/2, H/2, W/2]
        # x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)    # [B, 32, D/2, H/2, W/2]
        
        d2 = self.Up2(d3)        # [B, 32, D, H, W]
        # x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)    # [B, 32, D, H, W]
        
        out = self.outc(d2)  # [B, out_channels, D, H, W]
        return out


""" =========================== Net2: +DASPP =================================================="""
class ResUNet_M_DASPP(ResUNet3D_M):
    def __init__(self, in_channels=4, 
                 out_channels=4, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(ResUNet_M_DASPP, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            trilinear=trilinear)
        self.bottleneck = DenseASPP3D(f_list[3], f_list[3])
        
    def forward(self, x):
        return super().forward(x)

""" =========================== Net2: +DASPP + MADG ========================================"""
class ResUNet_M_MADG_DASPP(ResUNet_M_DASPP):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(ResUNet_M_MADG_DASPP, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            trilinear=trilinear)
        
        self.Att5 = MultiAxisDualAttnGate(F_g=f_list[3], F_l=f_list[3], F_inter=f_list[3]//2)
        self.Att4 = MultiAxisDualAttnGate(F_g=f_list[2], F_l=f_list[2], F_inter=f_list[2]//2)
        self.Att3 = MultiAxisDualAttnGate(F_g=f_list[1], F_l=f_list[1], F_inter=f_list[1]//2)
        self.Att2 = MultiAxisDualAttnGate(F_g=f_list[0], F_l=f_list[0], F_inter=f_list[0]//2)
        
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
    
""" =========================== Net3: +DASPP + MADG + DCLA ========================================"""
class ResUNet_M_MADG_DASPP_DCLA(ResUNet_M_MADG_DASPP):
    def __init__(self, 
                in_channels=4, 
                out_channels=4, 
                f_list=[32, 64, 128, 256], 
                trilinear=True
                ):
        super(ResUNet_M_MADG_DASPP_DCLA, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            trilinear=trilinear)
        
        self.dcla =DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16])
        
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
        
        x5 = self.dcla([x1, x2, x3, x4], x5)
        # Bottleneck
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
             
             
class ResUNet_M_DCLA(ResUNet3D_M):
    def __init__(self, 
                in_channels=4, 
                out_channels=4, 
                f_list=[32, 64, 128, 256], 
                trilinear=True
                ):
        super(ResUNet_M_DCLA, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            trilinear=trilinear)
                 
        self.dcla =DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, kernel_size=3)
        
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
        
        x5 = self.dcla([x1, x2, x3, x4], x5)
        # Bottleneck
        x5 = self.bottleneck(x5)      # [B, 256, D/16, H/16, W/16]
        
        # Decoder with Attention
        d5 = self.Up5(x5)               # [B, 256, D/8, H/8, W/8]
        # x4 = self.Att5(g=d5, x=x4)      # [B, 256, D/8, H/8, W/8]
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)    # [B, 128, D/8, H/8, W/8]
        
        d4 = self.Up4(d5)        # [B, 128, D/4, H/4, W/4]
        # x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)    # [B, 64, D/4, H/4, W/4]
        
        d3 = self.Up3(d4)        # [B, 64, D/2, H/2, W/2]
        # x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)    # [B, 32, D/2, H/2, W/2]
        
        d2 = self.Up2(d3)        # [B, 32, D, H, W]
        # x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)    # [B, 32, D, H, W]
        
        out = self.outc(d2)  # [B, out_channels, D, H, W]
        return out

class ResUNet_M_LKDCLA_13_11_9_7(ResUNet_M_DCLA):
    def __init__(self, 
                in_channels=4, 
                out_channels=4, 
                f_list=[32, 64, 128, 256], 
                trilinear=True
                ):
        super(ResUNet_M_LKDCLA_13_11_9_7, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            trilinear=trilinear)
                 
        self.dcla =DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, kernel_size=(13, 11, 9, 7))
        
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
        
        x5 = self.dcla([x1, x2, x3, x4], x5)
        # Bottleneck
        x5 = self.bottleneck(x5)      # [B, 256, D/16, H/16, W/16]
        
        # Decoder with Attention
        d5 = self.Up5(x5)               # [B, 256, D/8, H/8, W/8]
        # x4 = self.Att5(g=d5, x=x4)      # [B, 256, D/8, H/8, W/8]
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)    # [B, 128, D/8, H/8, W/8]
        
        d4 = self.Up4(d5)        # [B, 128, D/4, H/4, W/4]
        # x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)    # [B, 64, D/4, H/4, W/4]
        
        d3 = self.Up3(d4)        # [B, 64, D/2, H/2, W/2]
        # x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)    # [B, 32, D/2, H/2, W/2]
        
        d2 = self.Up2(d3)        # [B, 32, D, H, W]
        # x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)    # [B, 32, D, H, W]
        
        out = self.outc(d2)  # [B, out_channels, D, H, W]
        return out


class ResUNet_M_DCLA_AG(ResUNet_M_DCLA):
    def __init__(self, 
                in_channels=4, 
                out_channels=4, 
                f_list=[32, 64, 128, 256], 
                trilinear=True
                ):
        super(ResUNet_M_DCLA_AG, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            trilinear=trilinear)
                 
        self.Att5 = AttentionGate(F_g=f_list[3], F_l=f_list[3], F_inter=f_list[3]//2)
        self.Att4 = AttentionGate(F_g=f_list[2], F_l=f_list[2], F_inter=f_list[2]//2)
        self.Att3 = AttentionGate(F_g=f_list[1], F_l=f_list[1], F_inter=f_list[1]//2)
        self.Att2 = AttentionGate(F_g=f_list[0], F_l=f_list[0], F_inter=f_list[0]//2)
        
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
        
        x5 = self.dcla([x1, x2, x3, x4], x5)
        # Bottleneck
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

class ResUNet_M_DCLA_AG_DASPP(ResUNet_M_DCLA_AG):
    def __init__(self, 
                in_channels=4, 
                out_channels=4, 
                f_list=[32, 64, 128, 256], 
                trilinear=True
                ):
        super(ResUNet_M_DCLA_AG_DASPP, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            trilinear=trilinear)
        self.bottleneck = DenseASPP3D(f_list[3], f_list[3])
        
    def forward(self, x):
        return super().forward(x)

class ResUNet_M_DCLA_MADG(ResUNet_M_DCLA):
    def __init__(self, 
                in_channels=4, 
                out_channels=4, 
                f_list=[32, 64, 128, 256], 
                trilinear=True
                ):
        super(ResUNet_M_DCLA_MADG, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            trilinear=trilinear)
                 
        self.Att5 = MultiAxisDualAttnGate(F_g=f_list[3], F_l=f_list[3], F_inter=f_list[3]//2)
        self.Att4 = MultiAxisDualAttnGate(F_g=f_list[2], F_l=f_list[2], F_inter=f_list[2]//2)
        self.Att3 = MultiAxisDualAttnGate(F_g=f_list[1], F_l=f_list[1], F_inter=f_list[1]//2)
        self.Att2 = MultiAxisDualAttnGate(F_g=f_list[0], F_l=f_list[0], F_inter=f_list[0]//2)
        
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
        
        x5 = self.dcla([x1, x2, x3, x4], x5)
        # Bottleneck
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


class ResUNet_M_DCLA_MADG_DASPP(ResUNet_M_DCLA_MADG):
    def __init__(self, 
                in_channels=4, 
                out_channels=4, 
                f_list=[32, 64, 128, 256], 
                trilinear=True
                ):
        super(ResUNet_M_DCLA_MADG_DASPP, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            trilinear=trilinear)
        self.bottleneck = DenseASPP3D(f_list[3], f_list[3])
        
    def forward(self, x):
        return super().forward(x)


if __name__ == "__main__":
    test_unet(model_class=ResUNet_M_DCLA_MADG_DASPP, batch_size=1)   
        
             