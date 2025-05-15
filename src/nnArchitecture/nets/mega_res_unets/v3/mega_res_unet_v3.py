# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 12:30:32
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建 Mega Res UNe v3
*      VERSION: v3.0
*      FEATURES: 
                1. 使用大核关注太多局部信息会导致全局信息倍掩盖
                2. 但是使用小核会导致局部信息丢失
                3. 所以需要使用DASPP来聚合全局信息
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

from nnArchitecture.nets.mega_res_unets.v3 import (
    MegaResConv,
    ResConv3D_S_BN,
    DynamicCrossLevelAttention,
    AttentionGate,
    MultiAxisDualAttnGate,
    DenseASPP3D
)

from nnArchitecture.nets.mega_res_unets.v2 import (
    MegaResUNetv2,
    MegaResUNetv2_DCLA
)
    

from utils.test_unet import test_unet

class MegaResUNetv3(MegaResUNetv2):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(MegaResUNetv3, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            kernel_size=kernel_size,
            trilinear=trilinear)
        self.bottleneck = DenseASPP3D(in_channels=f_list[3], out_channels=f_list[3], dilations=[1, 2, 3])
        
    def forward(self, x):
        out = super().forward(x)
        return out
    
class MegaResUNetv3_DCLA(MegaResUNetv2_DCLA):
    def __init__(self, in_channels=4, out_channels=4, kernel_size=7, f_list=[32, 64, 128, 256], trilinear=True):
        super(MegaResUNetv2_DCLA, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            kernel_size=kernel_size,
            trilinear=trilinear)
        self.bottleneck = DenseASPP3D(in_channels=f_list[3], out_channels=f_list[3], dilations=[1, 2, 3])

    def forward(self, x):
        return super().forward(x)
    
class MegaResUNetv3_LKDCLA(MegaResUNetv3_DCLA):
    def __init__(self, in_channels=4, out_channels=4, kernel_size=7, f_list=[32, 64, 128, 256], trilinear=True):
        super(MegaResUNetv3_LKDCLA, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            kernel_size=kernel_size,
            trilinear=trilinear)
        self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, kernel_size=7)

    def forward(self, x):
        return super().forward(x)

class MegaResUNetv3_SLKDCLA(MegaResUNetv3_LKDCLA):
    def __init__(self, in_channels=4, out_channels=4, kernel_size=7, f_list=[32, 64, 128, 256], trilinear=True):
        super(MegaResUNetv3_SLKDCLA, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            kernel_size=kernel_size,
            trilinear=trilinear)
        self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, kernel_size=(11, 11, 11, 7))
    def forward(self, x):
        return super().forward(x)
    
if __name__ == "__main__":
    test_unet(model_class=MegaResUNetv3_SLKDCLA, batch_size=1)   