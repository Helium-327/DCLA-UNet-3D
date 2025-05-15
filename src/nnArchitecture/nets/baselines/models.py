'''
================================================
*      CREATE ON: 2025/04/30 17:14:40
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: DW ResUNet Baseline
*      VERSION: v1.0
*      FEATURES: 构建 DW ResUNet Baseline
*      STATE:    ✅ 
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


from nnArchitecture.commons import (
    init_weights_3d, 
    UpSample
)

from nnArchitecture.nets.baselines.mm import (
    DWResConv3D,
    ResConv3D_S_BN,
    ResConv3D_M_BN,
    ResConv3D,
    LightweightChannelAttention3Dv2,
    LightweightSpatialAttention3D,
    AttentionBlock3D
)

class DWResUNet(nn.Module):
    # 3.08 M
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True, dropout_rate=0):
        super(DWResUNet, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = DWResConv3D(in_channels, f_list[0], dropout_rate=dropout_rate)
        self.Conv2 = DWResConv3D(f_list[0], f_list[1], dropout_rate=dropout_rate)
        self.Conv3 = DWResConv3D(f_list[1], f_list[2], dropout_rate=dropout_rate)
        self.Conv4 = DWResConv3D(f_list[2], f_list[3], dropout_rate=dropout_rate)
        
        self.bottle_neck = DWResConv3D(f_list[3], f_list[3], dropout_rate=dropout_rate)
        
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.UpConv4 = DWResConv3D(f_list[3]*2, f_list[3]//2, dropout_rate=dropout_rate)
        
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.UpConv3 = DWResConv3D(f_list[2]*2, f_list[2]//2, dropout_rate=dropout_rate)
        
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.UpConv2 = DWResConv3D(f_list[1]*2, f_list[1]//2, dropout_rate=dropout_rate)
        
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        self.UpConv1 = DWResConv3D(f_list[0]*2, f_list[0], dropout_rate=dropout_rate)
        
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
    
        x5 = self.bottle_neck(x5)  # [B, 256, D/16, H/16, W/16]
        
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
    

    
class ResUNetBaseline_M(nn.Module):
    def __init__(self, in_channels=4, out_channels=4,f_list=[32, 64, 128, 256], trilinear=True, dropout_rate=0):
        super(ResUNetBaseline_M, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = ResConv3D_M_BN(in_channels, f_list[0])
        self.Conv2 = ResConv3D_M_BN(f_list[0], f_list[1])
        self.Conv3 = ResConv3D_M_BN(f_list[1], f_list[2])
        self.Conv4 = ResConv3D_M_BN(f_list[2], f_list[3])

        self.bottle_neck = ResConv3D_M_BN(f_list[3], f_list[3])
        
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.UpConv4 = ResConv3D_M_BN(f_list[3]*2, f_list[3]//2)
        
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.UpConv3 = ResConv3D_M_BN(f_list[2]*2, f_list[2]//2)
        
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.UpConv2 = ResConv3D_M_BN(f_list[1]*2, f_list[1]//2)
        
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        self.UpConv1 = ResConv3D_M_BN(f_list[0]*2, f_list[0])
        
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
        
        x5 = self.bottle_neck(x5)
    
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
    
class ResUNetBaseline_S(nn.Module):
    # 3.08 M
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True, dropout_rate=0):
        super(ResUNetBaseline_S, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = ResConv3D_S_BN(in_channels, f_list[0], dropout_rate=dropout_rate)
        self.Conv2 = ResConv3D_S_BN(f_list[0], f_list[1], dropout_rate=dropout_rate)
        self.Conv3 = ResConv3D_S_BN(f_list[1], f_list[2], dropout_rate=dropout_rate)
        self.Conv4 = ResConv3D_S_BN(f_list[2], f_list[3], dropout_rate=dropout_rate)
        
        self.bottle_neck = ResConv3D_S_BN(f_list[3], f_list[3], dropout_rate=dropout_rate)
        
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
    
        x5 = self.bottle_neck(x5)
        
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

    
class RA_UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True):
        super(RA_UNet, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.Conv1 = ResConv3D(in_channels, f_list[0])
        self.Conv2 = ResConv3D(f_list[0], f_list[1])
        self.Conv3 = ResConv3D(f_list[1], f_list[2])
        self.Conv4 = ResConv3D(f_list[2], f_list[3])
        
        self.bottleneck = ResConv3D(f_list[3], f_list[3])
        
        self.Up5 = UpSample(f_list[3], f_list[3], trilinear)
        self.Att5 = AttentionBlock3D(F_g=f_list[3], F_l=f_list[3], F_int=f_list[3]//2)
        self.UpConv5 = ResConv3D(f_list[3]*2, f_list[3]//2)
        
        self.Up4 = UpSample(f_list[2], f_list[2], trilinear)
        self.Att4 = AttentionBlock3D(F_g=f_list[2], F_l=f_list[2], F_int=f_list[2]//2)
        self.UpConv4 = ResConv3D(f_list[2]*2, f_list[2]//2)
        
        self.Up3 = UpSample(f_list[1], f_list[1], trilinear)
        self.Att3 = AttentionBlock3D(F_g=f_list[1], F_l=f_list[1], F_int=f_list[1]//2)
        self.UpConv3 = ResConv3D(f_list[1]*2, f_list[1]//2)
        
        self.Up2 = UpSample(f_list[0], f_list[0], trilinear)
        self.Att2 = AttentionBlock3D(F_g=f_list[0], F_l=f_list[0], F_int=f_list[0]//2)
        self.UpConv2 = ResConv3D(f_list[0]*2, f_list[0])
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)

        self.apply(init_weights_3d)
        
    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)       # [B, 32, D, H, W]
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
        d5 = self.Up5(x5)        # [B, 256, D/8, H/8, W/8]
        x4 = self.Att5(g=d5, x=x4)
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
    
