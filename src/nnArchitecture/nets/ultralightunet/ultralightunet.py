# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/26 17:56:07
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 复现UltralightUNet ,并拓展至3D
*      VERSION: v1.0
*      FEATURES: 
*      STATE:     
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

from nnArchitecture.nets.ultralightunet.modules import *

from utils.test_unet import test_unet


# class UltralightUNet(nn.Module): # SLKv3 + DCLA + MSF
#     __remark__ = """
#     [Version]: V1
#     [Author]: Junyin Xiong
#     # TODO: 
#     [Features]
#     • 总参数量: 638.617K
#     • FLOPs: 44.881G
#     [Changes]
#     [测试集结果]
    
#     [滑窗推理结果]
#     """
#     def __init__(self, 
#                  in_channels=4, 
#                  out_channels=4,
#                  kernel_size=3, 
#                  ch_list=[32, 64, 128, 256], 
#                  f_size=[128, 64, 32, 16],
#                  min_size=8,
#                  trilinear=True,
#                  dropout_rate=0,
#                  norm_type = 'batch',
#                  act_type = 'gelu',
#                  use_ag = False,
#                  ):
#         super(UltralightUNet, self).__init__()
#         self.use_ag = use_ag
#         # Max Pooling
#         self.add_module("maxpool", nn.MaxPool3d(kernel_size=2, stride=2))
        
#         # Encoders
#         self.add_module("encoder1", MultiKernelInvertedResidualBlock(in_channels, ch_list[0], [3,5,7], norm_type=norm_type, act_type=act_type))
#         self.add_module("encoder2", MultiKernelInvertedResidualBlock(ch_list[0], ch_list[1], [3,5,7], norm_type=norm_type, act_type=act_type))
#         self.add_module("encoder3", MultiKernelInvertedResidualBlock(ch_list[1], ch_list[2], [3,5,7], norm_type=norm_type, act_type=act_type))
#         self.add_module("encoder4", MultiKernelInvertedResidualBlock(ch_list[2], ch_list[3], [3,5,7], norm_type=norm_type, act_type=act_type))
        
#         # Dynamic Cross-Level Attention
#         # self.add_module("dcla", DynamicCrossLevelAttention(ch_list, f_size, min_size, 1, 7, 1)) 

#         # Upsampling
#         self.add_module("upsample4", UpSample(ch_list[3], ch_list[3], trilinear=trilinear))
#         self.add_module("upsample3", UpSample(ch_list[2], ch_list[2], trilinear=trilinear))
#         self.add_module("upsample2", UpSample(ch_list[1], ch_list[1], trilinear=trilinear))
#         self.add_module("upsample1", UpSample(ch_list[0], ch_list[0], trilinear=trilinear))
       
#        # Decoders 
#         self.add_module("decoder4", MultiKernelInvertedResidualBlock(ch_list[3]*2, ch_list[2], [3,5,7], norm_type=norm_type, act_type=act_type))
#         self.add_module("decoder3", MultiKernelInvertedResidualBlock(ch_list[2]*2, ch_list[1], [3,5,7], norm_type=norm_type, act_type=act_type))
#         self.add_module("decoder2", MultiKernelInvertedResidualBlock(ch_list[1]*2, ch_list[0], [3,5,7], norm_type=norm_type, act_type=act_type))
#         self.add_module("decoder1", MultiKernelInvertedResidualBlock(ch_list[0]*2, ch_list[0], [3,5,7], norm_type=norm_type, act_type=act_type))
       
#         # Attention Gate
#         self.add_module("gag4", GroupedAttentionGate(F_g=ch_list[3], F_l=ch_list[3], F_int=ch_list[3]//2, kernel_size=1, norm_type=norm_type, act_type=act_type, groups=1)) if use_ag else nn.Identity()
#         self.add_module("gag3", GroupedAttentionGate(F_g=ch_list[2], F_l=ch_list[2], F_int=ch_list[2]//2, kernel_size=1, norm_type=norm_type, act_type=act_type, groups=1)) if use_ag else nn.Identity()
#         self.add_module("gag2", GroupedAttentionGate(F_g=ch_list[1], F_l=ch_list[1], F_int=ch_list[1]//2, kernel_size=1, norm_type=norm_type, act_type=act_type, groups=1)) if use_ag else nn.Identity()
#         self.add_module("gag1", GroupedAttentionGate(F_g=ch_list[0], F_l=ch_list[0], F_int=ch_list[0]//2, kernel_size=1, norm_type=norm_type, act_type=act_type, groups=1)) if use_ag else nn.Identity()
        
#         self.add_module("outc", nn.Conv3d(ch_list[0], out_channels, kernel_size=1, stride=1, padding=0))
#         self.apply(init_weights_3d)        
        
        
#     def forward(self, x):
#         out = self.encoder1(x)              # [1,4,128,128,128] --> [1,32,128,128,128]
#         x1 = out                            # [1,32,128,128,128]
#         out = self.maxpool(out)             # [1,32,128,128,128] --> [1,32,64,64,64]
        
#         out = self.encoder2(out)            # [1,32,64,64,64] --> [1,64,64,64,64]
#         x2 = out                            # [1,64,64,64,64]   
#         out = self.maxpool(out)             # [1,64,64,64,64] --> [1,64,32,32,32]
    
#         out = self.encoder3(out)            # [1,64,32,32,32] --> [1,128,32,32,32]
#         x3 = out                            # [1,128,32,32,32]
#         out = self.maxpool(out)             # [1,128,32,32,32] --> [1,128,16,16,16]
        
#         out = self.encoder4(out)            # [1,128,16,16,16] --> [1,256,16,16,16]
#         x4 = out                            # [1,256,16,16,16]
#         out = self.maxpool(out)             # [1,256,16,16,16] --> [1,256,8,8,8]
        
#         out = self.dcla([x1, x2, x3, x4], out) if hasattr(self, 'dcla') else out   # [1,256,8,8,8]
        
#         out = self.upsample4(out)            # [1,256,8,8,8] --> [1,256,16,16,16]
#         out = self.gag4(out, x4) if self.use_ag else out           # [1,256,16,16,16]
#         out = torch.cat((out, x4), dim=1)    # [1,256+256,16,16,16] --> [1,512,16,16,16]
#         out = self.decoder4(out)             # [1,512,16,16,16] --> [1,128,16,16,16]
        
#         out = self.upsample3(out)            # [1,128,16,16,16] --> [1,128,32,32,32]
#         out = self.gag3(out, x3) if self.use_ag else out            # [1,128,32,32,32]
#         out = torch.cat((out, x3), dim=1)    # [1,128+128,32,32,32] --> [1,256,32,32,32]
#         out = self.decoder3(out)             # [1,256,32,32,32] --> [1,64,32,32,32]
        
#         out = self.upsample2(out)            # [1,64,32,32,32] --> [1,64,64,64,64]
#         out = self.gag2(out, x2) if self.use_ag else out            # [1,64,64,64,64]
#         out = torch.cat((out, x2), dim=1)    # [1,64+64,64,64,64] --> [1,128,64,64,64]
#         out = self.decoder2(out)             # [1,128,64,64,64] --> [1,32,64,64,64]
        
#         out = self.upsample1(out)            # [1,32,64,64,64] --> [1,32,128,128,128]
#         out = self.gag1(out, x1) if self.use_ag else out           # [1,32,128,128,128]
#         out = torch.cat((out, x1), dim=1)    # [1,32+32,128,128,128] --> [1,64,128,128,128]
#         out = self.decoder1(out)             # [1,64,128,128,128] --> [1,32,128,128,128]
        
#         out = self.outc(out)            # [1,32,128,128,128] --> [1,out_channels,128,128,128]
        
#         return out

def attn_mapping(sa, ca, x, mapping_type='mul'):
    """
    计算注意力映射
    :param sa: 空间注意力
    :param ca: 通道注意力
    :param x: 输入特征图
    :param mapping_type: 选择的注意力映射方式 add | mul
    :return: 注意力映射后的特征图
    
    """
    if mapping_type == 'add':
        return sa*x + ca*x
    elif mapping_type == 'mul':
        return sa * x * ca
    else:
        raise ValueError('Invalid mapping type')
    

class UltralightUNet(nn.Module): # SLKv3 + DCLA + MSF
    __remark__ = """
    [Version]: V1
    [Author]: Junyin Xiong
    # TODO: 
    [Features]
    • 总参数量: 638.617K
    • FLOPs: 44.881G
    [Changes]
    [测试集结果]
    
    [滑窗推理结果]
    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 kernel_size=3, 
                 ch_list=[32, 64, 128, 256], 
                 f_size=[128, 64, 32, 16],
                 min_size=8,
                 trilinear=True,
                 dropout_rate=0,
                 norm_type = 'batch',
                 act_type = 'relu6',
                 use_ag = True,
                 ):
        super(UltralightUNet, self).__init__()
        self.use_ag = use_ag
        # Max Pooling
        self.add_module("maxpool", nn.MaxPool3d(kernel_size=2, stride=2))
        
        # Encoders
        self.add_module("encoder1", DepthAxialMultiScaleResidualBlock(in_channels, ch_list[0], kernel_size, 1, [1,2,3], norm_type=norm_type, act_type=act_type))
        self.add_module("encoder2", DepthAxialMultiScaleResidualBlock(ch_list[0], ch_list[1], kernel_size, 1, [1,2,3], norm_type=norm_type, act_type=act_type))
        self.add_module("encoder3", DepthAxialMultiScaleResidualBlock(ch_list[1], ch_list[2], kernel_size, 1, [1,2,3], norm_type=norm_type, act_type=act_type))
        self.add_module("encoder4", DepthAxialMultiScaleResidualBlock(ch_list[2], ch_list[3], kernel_size, 1, [1,2,3], norm_type=norm_type, act_type=act_type))

        # self.add_module()
        
        # Dynamic Cross-Level Attention
        self.add_module("dcla", DynamicCrossLevelAttention(ch_list, f_size, min_size, 1, 7, 1, norm_type=norm_type, act_type=act_type)) 

        # Upsampling
        self.add_module("upsample4", UpSample(ch_list[3], ch_list[3], trilinear=trilinear))
        self.add_module("upsample3", UpSample(ch_list[2], ch_list[2], trilinear=trilinear))
        self.add_module("upsample2", UpSample(ch_list[1], ch_list[1], trilinear=trilinear))
        self.add_module("upsample1", UpSample(ch_list[0], ch_list[0], trilinear=trilinear))
       
       # Decoders 
        self.add_module("decoder4", DepthAxialMultiScaleResidualBlock(ch_list[3]*2, ch_list[2], kernel_size, 1, [1,2,3], norm_type=norm_type, act_type=act_type))
        self.add_module("decoder3", DepthAxialMultiScaleResidualBlock(ch_list[2]*2, ch_list[1], kernel_size, 1, [1,2,3], norm_type=norm_type, act_type=act_type))
        self.add_module("decoder2", DepthAxialMultiScaleResidualBlock(ch_list[1]*2, ch_list[0], kernel_size, 1, [1,2,3], norm_type=norm_type, act_type=act_type))
        self.add_module("decoder1", DepthAxialMultiScaleResidualBlock(ch_list[0]*2, ch_list[0], kernel_size, 1, [1,2,3], norm_type=norm_type, act_type=act_type))
       
        # Attention Gate
        self.add_module("gag4", GroupedAttentionGate(F_g=ch_list[3], F_l=ch_list[3], F_int=ch_list[3]//2, kernel_size=1, norm_type=norm_type, act_type=act_type, groups=1)) if use_ag else nn.Identity()
        self.add_module("gag3", GroupedAttentionGate(F_g=ch_list[2], F_l=ch_list[2], F_int=ch_list[2]//2, kernel_size=1, norm_type=norm_type, act_type=act_type, groups=1)) if use_ag else nn.Identity()
        self.add_module("gag2", GroupedAttentionGate(F_g=ch_list[1], F_l=ch_list[1], F_int=ch_list[1]//2, kernel_size=1, norm_type=norm_type, act_type=act_type, groups=1)) if use_ag else nn.Identity()
        self.add_module("gag1", GroupedAttentionGate(F_g=ch_list[0], F_l=ch_list[0], F_int=ch_list[0]//2, kernel_size=1, norm_type=norm_type, act_type=act_type, groups=1)) if use_ag else nn.Identity()
        
        # Channels Attention 
        self.CA1 = ChannelAttention(ch_list[0], ratio=16, act_type="swish")
        self.CA2 = ChannelAttention(ch_list[0], ratio=16, act_type="swish")
        self.CA3 = ChannelAttention(ch_list[1], ratio=16, act_type="swish")
        self.CA4 = ChannelAttention(ch_list[2], ratio=16, act_type="swish")
        
        # Spatial Attention 
        self.SA = SpatialAttention(kernel_size=7)
        
        self.add_module("outc", nn.Conv3d(ch_list[0], out_channels, kernel_size=1, stride=1, padding=0))
        self.apply(init_weights_3d)        
        
    def forward(self, x):
        out = self.encoder1(x)              # [1,4,128,128,128] --> [1,32,128,128,128]
        x1 = out                            # [1,32,128,128,128]
        out = self.maxpool(out)             # [1,32,128,128,128] --> [1,32,64,64,64]
        
        out = self.encoder2(out)            # [1,32,64,64,64] --> [1,64,64,64,64]
        x2 = out                            # [1,64,64,64,64]   
        out = self.maxpool(out)             # [1,64,64,64,64] --> [1,64,32,32,32]
    
        out = self.encoder3(out)            # [1,64,32,32,32] --> [1,128,32,32,32]
        x3 = out                            # [1,128,32,32,32]
        out = self.maxpool(out)             # [1,128,32,32,32] --> [1,128,16,16,16]
        
        out = self.encoder4(out)            # [1,128,16,16,16] --> [1,256,16,16,16]
        x4 = out                            # [1,256,16,16,16]
        out = self.maxpool(out)             # [1,256,16,16,16] --> [1,256,8,8,8]
        
        out = self.dcla([x1, x2, x3, x4], out) if hasattr(self, 'dcla') else out   # [1,256,8,8,8]
        
        out = self.upsample4(out)            # [1,256,8,8,8] --> [1,256,16,16,16]
        out = self.gag4(out, x4) if self.use_ag else out           # [1,256,16,16,16]
        out = torch.cat((out, x4), dim=1)    # [1,256+256,16,16,16] --> [1,512,16,16,16]
        out = self.decoder4(out)             # [1,512,16,16,16] --> [1,128,16,16,16]
        out = attn_mapping(self.SA(out), self.CA4(out), out, mapping_type='mul')

        out = self.upsample3(out)            # [1,128,16,16,16] --> [1,128,32,32,32]
        out = self.gag3(out, x3) if self.use_ag else out            # [1,128,32,32,32]
        out = torch.cat((out, x3), dim=1)    # [1,128+128,32,32,32] --> [1,256,32,32,32]
        out = self.decoder3(out)             # [1,256,32,32,32] --> [1,64,32,32,32]
        out = attn_mapping(self.SA(out), self.CA3(out), out, mapping_type='mul')
        
        out = self.upsample2(out)            # [1,64,32,32,32] --> [1,64,64,64,64]
        out = self.gag2(out, x2) if self.use_ag else out            # [1,64,64,64,64]
        out = torch.cat((out, x2), dim=1)    # [1,64+64,64,64,64] --> [1,128,64,64,64]
        out = self.decoder2(out)             # [1,128,64,64,64] --> [1,32,64,64,64]
        out = attn_mapping(self.SA(out), self.CA2(out), out, mapping_type='mul')
        
        out = self.upsample1(out)            # [1,32,64,64,64] --> [1,32,128,128,128]
        out = self.gag1(out, x1) if self.use_ag else out           # [1,32,128,128,128]
        out = torch.cat((out, x1), dim=1)    # [1,32+32,128,128,128] --> [1,64,128,128,128]
        out = self.decoder1(out)             # [1,64,128,128,128] --> [1,32,128,128,128]
        out = attn_mapping(self.SA(out), self.CA1(out), out, mapping_type='mul')
        out = self.outc(out)            # [1,32,128,128,128] --> [1,out_channels,128,128,128]
        
        return out
        
        
        
if __name__ == "__main__":
    test_unet(model_class=UltralightUNet, batch_size=1)   
    model = UltralightUNet(in_channels=4, out_channels=4)
    print(model.__remark__)
        
    
    
    