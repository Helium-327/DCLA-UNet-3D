# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 12:44:25
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建  Mega Res UNet v4
*      VERSION: v8.0
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

from nnArchitecture.nets.mega_res_unets.v7 import (
    MegaResUNetv7,
    MegaResUNetv7_LKDCLA
    
)

from utils.test_unet import test_unet


class MegaResUNetv8(MegaResUNetv7):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(MegaResUNetv8, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            kernel_size=kernel_size,
            trilinear=trilinear)
        
        self.Att5 = MultiAxisDualAttnGatev1(F_g=f_list[3], F_l=f_list[3], F_inter=f_list[3]//2)
        self.Att4 = MultiAxisDualAttnGatev1(F_g=f_list[2], F_l=f_list[2], F_inter=f_list[2]//2)
        self.Att3 = MultiAxisDualAttnGatev1(F_g=f_list[1], F_l=f_list[1], F_inter=f_list[1]//2)
        self.Att2 = MultiAxisDualAttnGatev1(F_g=f_list[0], F_l=f_list[0], F_inter=f_list[0]//2)
        
    def forward(self, x):
        return super().forward(x)
    
    
class MegaResUNetv8_LKDCLA(MegaResUNetv7_LKDCLA):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(MegaResUNetv8_LKDCLA, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            f_list=f_list, 
            kernel_size=kernel_size,
            trilinear=trilinear)
        
        self.Att5 = MultiAxisDualAttnGatev1(F_g=f_list[3], F_l=f_list[3], F_inter=f_list[3]//2)
        self.Att4 = MultiAxisDualAttnGatev1(F_g=f_list[2], F_l=f_list[2], F_inter=f_list[2]//2)
        self.Att3 = MultiAxisDualAttnGatev1(F_g=f_list[1], F_l=f_list[1], F_inter=f_list[1]//2)
        self.Att2 = MultiAxisDualAttnGatev1(F_g=f_list[0], F_l=f_list[0], F_inter=f_list[0]//2)
        
    def forward(self, x):
        return super().forward(x)
    
    
if __name__ == "__main__":
    test_unet(model_class=MegaResUNetv8, batch_size=1)   


