# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/01 18:43
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建 DCLA UNet v3 训练模型
*      VERSION: v3
*      FEATURES: 
               - ✅构建 DCLA UNet v3 系列模型结构
               - ✅编码器使用SLK特征提取模块。Kernel_size=5
               - ✅使用 DCLA 进行跨层特征融合，并对编码器进行增强，下采样核为7
               - ✅解码器使用MSF多尺度融合模块, Conv 1x1x1
               - ✅DCLA 下采样核为3,5,7
*      STATE:     开发中
*      CHANGE ON: 2025/05/03 22:43
=================================================
'''
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from nnArchitecture.commons import (
    init_weights_3d,
    UpSample
)
from nnArchitecture.commons import (
    DepthwiseAxialConv3d,
    LightweightChannelAttention3Dv2,
    LightweightSpatialAttention3D
)

from nnArchitecture.nets.dcla_nets.v4 import SLK, DCLCA, DCLSA,  MSF
from nnArchitecture.nets.dcla_nets.v4.baseline import ResUNetBaseline_S as baseline

from utils.test_unet import test_unet

class DCLA_UNet_v4(nn.Module):
    __remark__ = """
    [Version]: V4
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK大核卷积 (kernel_size=7)
    • 集成DCLA跨层注意力机制(Down_kernel=3, 5, 7) + 残差连接
    • 总参数量: 355.229K
    • FLOPs: 23.446G
    
    [测试集结果]

    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(DCLA_UNet_v4, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = SlimLargeKernelBlockv2(in_channels, f_list[0], kernel_size=kernel_size)
        self.Conv2 = SlimLargeKernelBlockv2(f_list[0], f_list[1], kernel_size=kernel_size)
        self.Conv3 = SlimLargeKernelBlockv2(f_list[1], f_list[2], kernel_size=kernel_size)
        self.Conv4 = SlimLargeKernelBlockv2(f_list[2], f_list[3], kernel_size=kernel_size)
        
        self.dcla1 =DCLSA(ch_list=[32], feats_size=[128], min_size=64, squeeze_kernel=1, down_kernel=[7], fusion_kernel=1)
        self.dcla2 =DCLSA(ch_list=[32, 64], feats_size=[128, 64], min_size=32, squeeze_kernel=1, down_kernel=[7], fusion_kernel=1)
        self.dcla3 =DCLSA(ch_list=[32, 64, 128], feats_size=[128, 64, 32], min_size=16, squeeze_kernel=1, down_kernel=[7], fusion_kernel=1)
        self.dcla4 =DCLSA(ch_list=[32, 64,128,256], feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=[7], fusion_kernel=1)
        
        self.ca1 = DCLCA(2*f_list[0], f_list[0])
        self.ca2 = DCLCA(f_list[0] + 2*f_list[1], f_list[1])
        self.ca3 = DCLCA(f_list[0] + f_list[1] + 2*f_list[2] , f_list[2])
        self.ca4 = DCLCA(f_list[0] + f_list[1] + f_list[2] + 2*f_list[3], f_list[3])
        
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = MutilScaleFusionBlockv2(in_channels=f_list[3]*2, out_channels=f_list[3]//2, fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        self.UpConv3 = MutilScaleFusionBlockv2(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        self.UpConv2 = MutilScaleFusionBlockv2(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        self.UpConv1 = MutilScaleFusionBlockv2(in_channels=f_list[0]*2, out_channels=f_list[0],     fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
        self.apply(init_weights_3d)  # 初始化权重
        
    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)                      # [B, 32, D, H, W]
        x1_d = self.MaxPool(x1)                 # [B, 32, D/2, H/2, W/2]
        x1_a = self.dcla1([x1], x1_d)           # [B, 32, D/2, H/2, W/2]
        x1_a = self.ca1([x1, x1_a], x1_a)       # [B, 32, D/2, H/2, W/2]
        
        x2 = self.Conv2(x1_a)                   # [B, 64, D/2, H/2, W/2]
        x2_d = self.MaxPool(x2)                 # [B, 64, D/4, H/4, W/4]
        x2_a = self.dcla2([x1, x2], x2_d)       # [B, 64, D/4, H/4, W/4]
        x2_a = self.ca2([x1, x2, x2_a], x2_a)   # [B, 64, D/4, H/4, W/4]
        
        x3 = self.Conv3(x2_a)      # [B, 128, D/4, H/4, W/4]
        x3_d = self.MaxPool(x3)
        x3_a = self.dcla3([x1, x2, x3], x3_d)
        x3_a = self.ca3([x1, x2, x3, x3_a], x3_a)
        
        x4 = self.Conv4(x3_a)      # [B, 256, D/8, H/8, W/8]
        x4_d = self.MaxPool(x4)
        x4_a = self.dcla4([x1, x2, x3, x4], x4_d)  # [B, 256, D/8, H/8, W/8]
        x4_a = self.ca4([x1, x2, x3, x4, x4_a], x4_a)
        
        # Decoder with Attention
        d5 = self.Up4(x4_a)               # [B, 256, D/8, H/8, W/8]
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
    
    
class ResUNetBaseline_S_SLK_MSF_v4(nn.Module):
    __remark__ = """
    [Version]: V4
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK大核卷积 (kernel_size=7)
    • 集成DCLA跨层注意力机制(Down_kernel=3, 5, 7) + 残差连接
    • 总参数量: 355.229K
    • FLOPs: 23.446G
    
    [测试集结果]

    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(ResUNetBaseline_S_SLK_MSF_v4, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = SlimLargeKernelBlockv2(in_channels, f_list[0], kernel_size=kernel_size)
        self.Conv2 = SlimLargeKernelBlockv2(f_list[0], f_list[1], kernel_size=kernel_size)
        self.Conv3 = SlimLargeKernelBlockv2(f_list[1], f_list[2], kernel_size=kernel_size)
        self.Conv4 = SlimLargeKernelBlockv2(f_list[2], f_list[3], kernel_size=kernel_size)
        
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = MutilScaleFusionBlockv2(in_channels=f_list[3]*2, out_channels=f_list[3]//2, fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        self.UpConv3 = MutilScaleFusionBlockv2(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        self.UpConv2 = MutilScaleFusionBlockv2(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        self.UpConv1 = MutilScaleFusionBlockv2(in_channels=f_list[0]*2, out_channels=f_list[0],     fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
        self.apply(init_weights_3d)  # 初始化权重
        
    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)                # [B, 32, D, H, W]
        x2 = self.MaxPool(x1)
        # x2 = self.dcla1([x1], x2)
        # x2 = self.ca1(x2)
        
        x2 = self.Conv2(x2)      # [B, 64, D/2, H/2, W/2]
        x3 = self.MaxPool(x2)
        # x3 = self.dcla2([x1, x2], x3)
        # x3 = self.ca2(x3)
        
        x3 = self.Conv3(x3)      # [B, 128, D/4, H/4, W/4]
        x4 = self.MaxPool(x3)
        # x4 = self.dcla3([x1, x2, x3], x4)
        # x4 = self.ca3(x4)
        
        x4 = self.Conv4(x4)      # [B, 256, D/8, H/8, W/8]
        x5 = self.MaxPool(x4)
        # x5 = self.dcla4([x1, x2, x3, x4], x5)  # [B, 256, D/8, H/8, W/8]
        # x5 = self.ca4(x5)
        
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
    
class ResUNetBaseline_S_SLK_DCLSA_MSF_v4(nn.Module):
    __remark__ = """
    [Version]: V4
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK大核卷积 (kernel_size=7)
    • 集成DCLA跨层注意力机制(Down_kernel=3, 5, 7) + 残差连接
    • 总参数量: 355.229K
    • FLOPs: 23.446G
    
    [测试集结果]

    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(ResUNetBaseline_S_SLK_DCLSA_MSF_v4, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = SlimLargeKernelBlockv2(in_channels, f_list[0], kernel_size=kernel_size)
        self.Conv2 = SlimLargeKernelBlockv2(f_list[0], f_list[1], kernel_size=kernel_size)
        self.Conv3 = SlimLargeKernelBlockv2(f_list[1], f_list[2], kernel_size=kernel_size)
        self.Conv4 = SlimLargeKernelBlockv2(f_list[2], f_list[3], kernel_size=kernel_size)
        self.dclsa1 =DCLSA(ch_list=[32], feats_size=[128], min_size=64, squeeze_kernel=1, down_kernel=[7], fusion_kernel=7)
        self.dclsa2 =DCLSA(ch_list=[32, 64], feats_size=[128, 64], min_size=32, squeeze_kernel=1, down_kernel=[7], fusion_kernel=7)
        self.dclsa3 =DCLSA(ch_list=[32, 64, 128], feats_size=[128, 64, 32], min_size=16, squeeze_kernel=1, down_kernel=[7], fusion_kernel=7)
        self.dclsa4 =DCLSA(ch_list=[32, 64,128,256], feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=[7], fusion_kernel=7)
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = MutilScaleFusionBlockv2(in_channels=f_list[3]*2, out_channels=f_list[3]//2, fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        self.UpConv3 = MutilScaleFusionBlockv2(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        self.UpConv2 = MutilScaleFusionBlockv2(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        self.UpConv1 = MutilScaleFusionBlockv2(in_channels=f_list[0]*2, out_channels=f_list[0],     fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
        self.apply(init_weights_3d)  # 初始化权重
        
    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)                      # [B, 32, D, H, W]
        x1_d = self.MaxPool(x1)                 # [B, 32, D/2, H/2, W/2]
        x1_a = self.dclsa1([x1], x1_d)           # [B, 32, D/2, H/2, W/2]
        # x1_a = self.dclca1([x1, x1_a], x1_a)       # [B, 32, D/2, H/2, W/2]
        
        x2 = self.Conv2(x1_a)                   # [B, 64, D/2, H/2, W/2]
        x2_d = self.MaxPool(x2)                 # [B, 64, D/4, H/4, W/4]
        x2_a = self.dclsa2([x1, x2], x2_d)       # [B, 64, D/4, H/4, W/4]
        # x2_a = self.dclca2([x1, x2, x2_a], x2_a)   # [B, 64, D/4, H/4, W/4]
        
        x3 = self.Conv3(x2_a)      # [B, 128, D/4, H/4, W/4]
        x3_d = self.MaxPool(x3)
        x3_a = self.dclsa3([x1, x2, x3], x3_d)
        # x3_a = self.dclca3([x1, x2, x3, x3_a], x3_a)
        
        x4 = self.Conv4(x3_a)      # [B, 256, D/8, H/8, W/8]
        x4_d = self.MaxPool(x4)
        x4_a = self.dclsa4([x1, x2, x3, x4], x4_d)  # [B, 256, D/8, H/8, W/8]
        # x4_a = self.dclca4([x1, x2, x3, x4, x4_a], x4_a)
        
        # Decoder with Attention
        d5 = self.Up4(x4_a)               # [B, 256, D/8, H/8, W/8]
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
    
    
class ResUNetBaseline_S_SLK_DCLCA_MSF_v4(nn.Module):
    __remark__ = """
    [Version]: V4
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK大核卷积 (kernel_size=7)
    • 集成DCLA跨层注意力机制(Down_kernel=3, 5, 7) + 残差连接
    • 总参数量: 355.229K
    • FLOPs: 23.446G
    
    [测试集结果]

    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(ResUNetBaseline_S_SLK_DCLCA_MSF_v4, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = SlimLargeKernelBlockv2(in_channels, f_list[0], kernel_size=kernel_size)
        self.Conv2 = SlimLargeKernelBlockv2(f_list[0], f_list[1], kernel_size=kernel_size)
        self.Conv3 = SlimLargeKernelBlockv2(f_list[1], f_list[2], kernel_size=kernel_size)
        self.Conv4 = SlimLargeKernelBlockv2(f_list[2], f_list[3], kernel_size=kernel_size)
        self.dclca1 = DCLCA(2*f_list[0], f_list[0])
        self.dclca2 = DCLCA(f_list[0] + 2*f_list[1], f_list[1])
        self.dclca3 = DCLCA(f_list[0] + f_list[1] + 2*f_list[2] , f_list[2])
        self.dclca4 = DCLCA(f_list[0] + f_list[1] + f_list[2] + 2*f_list[3], f_list[3])
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = MutilScaleFusionBlockv2(in_channels=f_list[3]*2, out_channels=f_list[3]//2, fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        self.UpConv3 = MutilScaleFusionBlockv2(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        self.UpConv2 = MutilScaleFusionBlockv2(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        self.UpConv1 = MutilScaleFusionBlockv2(in_channels=f_list[0]*2, out_channels=f_list[0],     fusion_kernel=kernel_size, use_attn=True, use_fusion=True)
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
        self.apply(init_weights_3d)  # 初始化权重
        
    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)                      # [B, 32, D, H, W]
        x1_d = self.MaxPool(x1)                 # [B, 32, D/2, H/2, W/2]
        # x1_a = self.dclsa1([x1], x1_d)           # [B, 32, D/2, H/2, W/2]
        x1_a = self.dclca1([x1, x1_d], x1_d)       # [B, 32, D/2, H/2, W/2]
        
        x2 = self.Conv2(x1_a)                   # [B, 64, D/2, H/2, W/2]
        x2_d = self.MaxPool(x2)                 # [B, 64, D/4, H/4, W/4]
        # x2_a = self.dclsa2([x1, x2], x2_d)       # [B, 64, D/4, H/4, W/4]
        x2_a = self.dclca2([x1, x2, x2_d], x2_d)   # [B, 64, D/4, H/4, W/4]
        
        x3 = self.Conv3(x2_a)      # [B, 128, D/4, H/4, W/4]
        x3_d = self.MaxPool(x3)
        # x3_a = self.dclsa3([x1, x2, x3], x3_d)
        x3_a = self.dclca3([x1, x2, x3, x3_d], x3_d)
        
        x4 = self.Conv4(x3_a)      # [B, 256, D/8, H/8, W/8]
        x4_d = self.MaxPool(x4)
        # x4_a = self.dclsa4([x1, x2, x3, x4], x4_d)  # [B, 256, D/8, H/8, W/8]
        x4_a = self.dclca4([x1, x2, x3, x4, x4_d], x4_d)
        
        # Decoder with Attention
        d5 = self.Up4(x4_a)               # [B, 256, D/8, H/8, W/8]
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
# class ECA(nn.Module):
#     def __init__(self, in_channels, out_channels, ratio=2):
#         super(ECA, self).__init__()
        
#         self.gap = nn.AdaptiveAvgPool3d(1)
#         self.ca = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels//ratio,  # 每个分支的输出通道数为总输出通道数的一半
#                 kernel_size=1
#             ),
#             nn.GELU(),
#             nn.Conv3d(
#                 out_channels//ratio,
#                 out_channels,  # 每个分支的输出通道数为总输出通道数的一半
#                 kernel_size=1
#             ),
#             nn.Sigmoid()
#         )
        
#     def forward(self, feats, x):
#         feats_gap = [self.gap(feat) for feat in feats]
#         feats_cat = torch.cat(feats_gap, dim=1)
#         attn = self.ca(feats_cat)
#         return x * attn + x
    
# class ECA(nn.Module):
#     def __init__(self, in_channels, out_channels, ratio=2):
#         super(ECA, self).__init__()
        
#         self.gap = nn.AdaptiveAvgPool3d(1)
#         self.gmp = nn.AdaptiveAvgPool3d(1)
#         self.ca = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels//ratio,  # 每个分支的输出通道数为总输出通道数的一半
#                 kernel_size=1
#             ),
#             nn.GELU(),
#             nn.Conv3d(
#                 out_channels//ratio,
#                 out_channels,  # 每个分支的输出通道数为总输出通道数的一半
#                 kernel_size=1
#             ),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         feats_gap = self.gap(x) + self.gmp(x)
#         attn = self.ca(feats_gap)
#         return x * attn + x
    
class SlimLargeKernelBlockv2(nn.Module): 
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 ratio=4 
                ):
        super().__init__()
        self.depthwise =nn.Sequential(
                DepthwiseAxialConv3d(
                    in_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size
                ),
                nn.BatchNorm3d(out_channels),
                nn.GELU(),
                nn.Conv3d(
                    out_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=1
                ),
                nn.BatchNorm3d(out_channels),
            )                                       
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.GELU()
        
    def forward(self, x):
        out = self.depthwise(x)
        
        out += self.residual(x)
        return self.act(out)
    
class MutilScaleFusionBlockv2(nn.Module): #(MLP)
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                #  mid_channels=None,
                 kernel_size=3, 
                 dilations=[1,2,3], 
                 fusion_kernel=7,  # 尺度和编码器保持一致
                 ratio=2,               # 4比2好
                 act_op='gelu',
                 use_attn=True,
                 use_fusion=True
                 ):
        """Multi kernel separable convolution fusion block
        
        Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int, optional): 卷积核大小. 默认为 3
        stride (int, optional): 卷积步长. 默认为 1
        dilations (list, optional): 空洞卷积的扩张率列表. 默认为 [1,2,3]
        groups (int, optional): 分组卷积的组数. 默认为 None
        """
        super().__init__()
        
        self.use_attn = use_attn
        self.use_fusion = use_fusion
        # 创建多个具有不同核的分离卷积分支
        self.sep_branchs = nn.ModuleList([
            nn.Sequential(
                DepthwiseAxialConv3d(
                    in_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size, 
                    dilation=d
                ),
                nn.BatchNorm3d(out_channels),
                nn.GELU()
            ) for d in dilations
        ])
        
        if use_attn:
            self.ca = LightweightChannelAttention3Dv2(kernel_size=7)
            self.sa = LightweightSpatialAttention3D(kernel_size=fusion_kernel)
            
        if use_fusion: 
            self.fusion = nn.Sequential(
                nn.Conv3d(
                    out_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=1
                ),
                nn.BatchNorm3d(out_channels)
            ) 
        # 残差连接，如果输入输出通道数不同，使用1x1卷积调整；否则使用恒等映射
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # relu
        self.act = nn.ReLU() if act_op == 'relu' else nn.GELU()
        
    def forward(self, x):
        out = x
        out = torch.sum(torch.stack([sep(out) for sep in self.sep_branchs]), dim=0)  # 使用加法而不是拼接
        
        if self.use_attn:
            channels_attn = self.ca(out)
            spatial_attn = self.sa(out)
            out = out * channels_attn * spatial_attn
            
        if self.use_fusion:
            out = self.fusion(out)

        out += self.residual(x)
        return self.act(out)   




if __name__ == "__main__":
    test_unet(model_class=ResUNetBaseline_S_SLK_DCLCA_MSF_v4, batch_size=1)   
    model = ResUNetBaseline_S_SLK_DCLCA_MSF_v4(in_channels=4, out_channels=4)
    print(model.__remark__)