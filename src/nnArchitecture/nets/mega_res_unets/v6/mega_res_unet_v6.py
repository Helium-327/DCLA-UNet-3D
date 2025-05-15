# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 12:44:25
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建  Mega Res UNet v4
*      VERSION: v6.0
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

from nnArchitecture.nets.mega_res_unets.v6 import (
    MegaResConv2,
    ResConv3D_S_BN,
    DynamicCrossLevelAttention,
    AttentionGate,
    MultiAxisDualAttnGate,
    DenseASPP3D
)

from nnArchitecture.nets.mega_res_unets.v5 import (
    MegaResUNetv5,
    MegaResUNetv5_DCLA,
    MegaResUNetv5_LKDCLA
    
)

from utils.test_unet import test_unet

class MegaResUNetv6(MegaResUNetv5):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(MegaResUNetv6, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            kernel_size=kernel_size,
            trilinear=trilinear)
        
        self.Conv1 = MegaResConv2(in_channels, f_list[0], kernel_size=kernel_size)
        self.Conv2 = MegaResConv2(f_list[0], f_list[1], kernel_size=kernel_size)
        self.Conv3 = MegaResConv2(f_list[1], f_list[2], kernel_size=kernel_size) 
        self.Conv4 = MegaResConv2(f_list[2], f_list[3], kernel_size=kernel_size)
        
        self.bottleneck = DenseASPP3D(in_channels=f_list[3], out_channels=f_list[3], dilations=[1, 2, 3])
        
        self.UpConv2 = MegaResConv2(f_list[0]*2, f_list[0], kernel_size=kernel_size)
        self.UpConv3 = MegaResConv2(f_list[1]*2, f_list[1]//2, kernel_size=kernel_size)
        self.UpConv4 = MegaResConv2(f_list[2]*2, f_list[2]//2, kernel_size=kernel_size)
        self.UpConv5 = MegaResConv2(f_list[3]*2, f_list[3]//2, kernel_size=kernel_size)
        
    def forward(self, x):
        return super().forward(x)
    
class MegaResUNetv6_DCLA(MegaResUNetv5_DCLA):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(MegaResUNetv6_DCLA, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            kernel_size=kernel_size,
            trilinear=trilinear)
        self.Conv1 = MegaResConv2(in_channels, f_list[0], kernel_size=kernel_size)
        self.Conv2 = MegaResConv2(f_list[0], f_list[1], kernel_size=kernel_size)
        self.Conv3 = MegaResConv2(f_list[1], f_list[2], kernel_size=kernel_size) 
        self.Conv4 = MegaResConv2(f_list[2], f_list[3], kernel_size=kernel_size)
        self.dcla =DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, kernel_size=3)
        self.bottleneck = DenseASPP3D(in_channels=f_list[3], out_channels=f_list[3], dilations=[1, 2, 3])
        
        self.UpConv2 = MegaResConv2(f_list[0]*2, f_list[0], kernel_size=kernel_size)
        self.UpConv3 = MegaResConv2(f_list[1]*2, f_list[1]//2, kernel_size=kernel_size)
        self.UpConv4 = MegaResConv2(f_list[2]*2, f_list[2]//2, kernel_size=kernel_size)
        self.UpConv5 = MegaResConv2(f_list[3]*2, f_list[3]//2, kernel_size=kernel_size)

    def forward(self, x):
        return super().forward(x)
    
class MegaResUNetv6_LKDCLA(MegaResUNetv6_DCLA):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(MegaResUNetv6_LKDCLA, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            kernel_size=kernel_size,
            trilinear=trilinear)
        self.dcla =DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, kernel_size=7)

    def forward(self, x):
        return super().forward(x)

class MegaResUNetv6_SLKDCLA(MegaResUNetv6_LKDCLA):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(MegaResUNetv6_SLKDCLA, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            kernel_size=kernel_size,
            trilinear=trilinear)
        self.dcla =DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, kernel_size=(11, 11, 11, 7))

    def forward(self, x):
        return super().forward(x)

if __name__ == "__main__":
    test_unet(model_class=MegaResUNetv6_SLKDCLA, batch_size=1)   


