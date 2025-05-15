# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 14:00:47
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建 Slim Sep UNet v1
*      VERSION: v2.0
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
from nnArchitecture.nets.slim_sep_unets.v1 import (
    SlimSepUNetv1,
    SlimSepUNetv1_LKDCLA
)

from nnArchitecture.nets.slim_sep_unets.v2 import (
    DenseASPP3D as DenseASPP3D_Lite,
    AttentionGate,
    SlimDownBlock,
    MultiKernelSlimUpBlockv2,
    DynamicCrossLevelAttentionv2,
    MultiAxisDualAttnGatev0
)

from utils.test_unet import test_unet


class SlimSepUNetv2(SlimSepUNetv1):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(SlimSepUNetv2, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,    
            f_list=f_list,
            trilinear=trilinear
        )
        self.UpConv5 = MultiKernelSlimUpBlockv2(f_list[3]*2, f_list[3]//2, kernel_size=3, dilations=[1, 2, 3]) # @1
        self.UpConv4 = MultiKernelSlimUpBlockv2(f_list[2]*2, f_list[2]//2, kernel_size=3, dilations=[1, 2, 3])
        self.UpConv3 = MultiKernelSlimUpBlockv2(f_list[1]*2, f_list[1]//2, kernel_size=3, dilations=[1, 2, 3])
        self.UpConv2 = MultiKernelSlimUpBlockv2(f_list[0]*2, f_list[0], kernel_size=3, dilations=[1, 2, 3])
        
    def forward(self, x):
        return super().forward(x)
    

class SlimSepUNetv2_LKDCLA(SlimSepUNetv1_LKDCLA):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(SlimSepUNetv2_LKDCLA, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,    
            f_list=f_list,
            trilinear=trilinear
        )
        self.dcla = DynamicCrossLevelAttentionv2(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, kernel_size=7) # @2
        self.UpConv5 = MultiKernelSlimUpBlockv2(f_list[3]*2, f_list[3]//2, kernel_size=3, dilations=[1, 2, 3])
        self.UpConv4 = MultiKernelSlimUpBlockv2(f_list[2]*2, f_list[2]//2, kernel_size=3, dilations=[1, 2, 3])
        self.UpConv3 = MultiKernelSlimUpBlockv2(f_list[1]*2, f_list[1]//2, kernel_size=3, dilations=[1, 2, 3])
        self.UpConv2 = MultiKernelSlimUpBlockv2(f_list[0]*2, f_list[0], kernel_size=3, dilations=[1, 2, 3])

    def forward(self, x):
        return super().forward(x)
    
class SlimSepUNetv2_LiteDASPP(SlimSepUNetv2):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(SlimSepUNetv2_LiteDASPP, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,    
            f_list=f_list,
            trilinear=trilinear
        )
        self.bottleneck = DenseASPP3D_Lite(in_channels=f_list[3], out_channels=f_list[3], dilations=[1, 2, 3])
        
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
    
# @2 在DCLAv2中引入残差
class SlimSepUNetv2_LiteDASPP_LKDCLA(SlimSepUNetv2_LiteDASPP):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(SlimSepUNetv2_LiteDASPP_LKDCLA, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,    
            f_list=f_list,
            trilinear=trilinear
        )
        self.dcla = DynamicCrossLevelAttentionv2(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, kernel_size=7) # @2
        self.UpConv5 = MultiKernelSlimUpBlockv2(f_list[3]*2, f_list[3]//2, kernel_size=3, dilations=[1, 2, 3])
        self.UpConv4 = MultiKernelSlimUpBlockv2(f_list[2]*2, f_list[2]//2, kernel_size=3, dilations=[1, 2, 3])
        self.UpConv3 = MultiKernelSlimUpBlockv2(f_list[1]*2, f_list[1]//2, kernel_size=3, dilations=[1, 2, 3])
        self.UpConv2 = MultiKernelSlimUpBlockv2(f_list[0]*2, f_list[0], kernel_size=3, dilations=[1, 2, 3])

    def forward(self, x):
        return super().forward(x)

class SlimSepUNetv2_LiteDASPP_LKDCLA_AG(SlimSepUNetv2_LiteDASPP_LKDCLA):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(SlimSepUNetv2_LiteDASPP_LKDCLA_AG, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,    
            f_list=f_list,
            trilinear=trilinear
        )
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

class SlimSepUNetv2_LiteDASPP_LKDCLA_MADG(SlimSepUNetv2_LiteDASPP_LKDCLA_AG):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(SlimSepUNetv2_LiteDASPP_LKDCLA_MADG, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,    
            f_list=f_list,
            trilinear=trilinear
        )
        self.Att5 = MultiAxisDualAttnGatev0(F_g=f_list[3], F_l=f_list[3], F_inter=f_list[3]//2)
        self.Att4 = MultiAxisDualAttnGatev0(F_g=f_list[2], F_l=f_list[2], F_inter=f_list[2]//2)
        self.Att3 = MultiAxisDualAttnGatev0(F_g=f_list[1], F_l=f_list[1], F_inter=f_list[1]//2)
        self.Att2 = MultiAxisDualAttnGatev0(F_g=f_list[0], F_l=f_list[0], F_inter=f_list[0]//2)
        
    def forward(self, x):
        return super().forward(x)

if __name__ == "__main__":
    test_unet(model_class=SlimSepUNetv2_LiteDASPP_LKDCLA_AG, batch_size=1)   