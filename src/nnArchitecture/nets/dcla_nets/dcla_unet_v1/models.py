# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/04/30 17:23:32
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建 DCLA UNet v1 ，并测试v1结构的有效性
*      VERSION: v1.0
*      FEATURES: 
               - ✅构建 DCLA UNet v1 模型结构
               - ✅编码器使用SLK特征提取模块
               - ✅使用 DCLA 进行跨层特征融合，并对编码器进行增强，下采样核为7
               - ✅解码器使用MSF多尺度融合模块
               - ✅DCLA后添加残差结构
            
*      STATE:     测试完成吗，效果有提升，但是指标未达预期（0.857）
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

from nnArchitecture.nets.dcla_nets.dcla_unet_v1.mm import (
    ResConv3D_S_BN,
    SlimLargeKernelBlock as SLK,
    MutilScaleFusionBlock as MSF,
    DynamicCrossLevelAttention as DCLA
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

class DCLA_UNet_v1(nn.Module):
    __remark__ = """
    [Version]: V1
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK (Conv1x1x1 + DWAC7x7x7)
    • 集成DCLA跨层注意力机制(Down_Kernel=3)
    • 解码器采用MSF多尺度融合模块(融合层用1x1)
    • 总参数量: 900.829K
    • FLOPs: 45.496G
    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True,
                 use_attn=True,
                 use_fusion=True
                 ):
        super(DCLA_UNet_v1, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = SLK(in_channels, f_list[0], kernel_size=7)
        self.Conv2 = SLK(f_list[0], f_list[1], kernel_size=7)
        self.Conv3 = SLK(f_list[1], f_list[2], kernel_size=7)
        self.Conv4 = SLK(f_list[2], f_list[3], kernel_size=7)
        
        self.dcla = DCLA(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=3, fusion_kernel=1)
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = MSF(in_channels=f_list[3]*2, out_channels=f_list[3]//2,  mid_channels=f_list[3],  use_attn=use_attn, use_fusion=use_fusion, fusion_kernel=7)
        self.UpConv3 = MSF(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  mid_channels=f_list[2], use_attn=use_attn, use_fusion=use_fusion, fusion_kernel=7)
        self.UpConv2 = MSF(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  mid_channels=f_list[1], use_attn=use_attn, use_fusion=use_fusion, fusion_kernel=7)
        self.UpConv1 = MSF(in_channels=f_list[0]*2, out_channels=f_list[0],  mid_channels=f_list[0], use_attn=use_attn, use_fusion=use_fusion, fusion_kernel=7)
        
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
    
        x5 = self.dcla([x1, x2, x3, x4], x5)  # [B, 256, D/8, H/8, W/8]
        
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
    
class ResUNetBaseline_S_SLK_v1(ResUNetBaseline_S):
    __remark__ = """
    [Version]: V1
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 
    • FLOPs: 
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0
                 ):
        super(ResUNetBaseline_S_SLK_v1, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate
        )
        self.Conv1 = SLK(in_channels, f_list[0], kernel_size=3)
        self.Conv2 = SLK(f_list[0], f_list[1], kernel_size=3)
        self.Conv3 = SLK(f_list[1], f_list[2], kernel_size=3)
        self.Conv4 = SLK(f_list[2], f_list[3], kernel_size=3)
    
    def forward(self, x):
        return super().forward(x)
    
class ResUNetBaseline_S_DCLA_SLK_v1(ResUNetBaseline_S_SLK_v1):
    __remark__ = """
    [Version]: V1
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 
    • FLOPs: 
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0
                 ):
        super(ResUNetBaseline_S_DCLA_SLK_v1, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate
        )
        self.dcla = DCLA(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=3, fusion_kernel=1)
        
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
    
        x5 = self.dcla([x1, x2, x3, x4], x5)  # [B, 256, D/8, H/8, W/8]
        
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

    
class ResUNetBaseline_S_MSF_v1(ResUNetBaseline_S):
    __remark__ = """
    [Version]: V1
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 
    • FLOPs: 
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0
                 ):
        super(ResUNetBaseline_S_MSF_v1, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate
        )
        
        self.UpConv4 = MSF(in_channels=f_list[3]*2, out_channels=f_list[3]//2)
        self.UpConv3 = MSF(in_channels=f_list[2]*2, out_channels=f_list[2]//2)
        self.UpConv2 = MSF(in_channels=f_list[1]*2, out_channels=f_list[1]//2)
        self.UpConv1 = MSF(in_channels=f_list[0]*2, out_channels=f_list[0])
        
    def forward(self, x):
        return super().forward(x)
    
class ResUNetBaseline_S_DCLA_MSF_v1(ResUNetBaseline_S_MSF_v1):
    __remark__ = """
    [Version]: V1
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 
    • FLOPs: 
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(ResUNetBaseline_S_DCLA_MSF_v1, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear
        )
        self.dcla = DCLA(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=3, fusion_kernel=1)
        
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
    
        x5 = self.dcla([x1, x2, x3, x4], x5)  # [B, 256, D/8, H/8, W/8]
        
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
    
class ResUNetBaseline_S_SLK_MSF_v1(DCLA_UNet_v1):
    __remark__ = """
    [Version]: V1
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 
    • FLOPs: 
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(ResUNetBaseline_S_SLK_MSF_v1, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear
        )
        if hasattr(self, 'dcla'):
            delattr(self, 'dcla')        
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
    
class ResUNetBaseline_S_DCLA_v1(ResUNetBaseline_S):
    __remark__ = """
    [Version]: V1
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 
    • FLOPs: 
    """
    def __init__(self,
                    in_channels=4,
                    out_channels=4,
                    f_list=[32, 64, 128, 256],
                    trilinear=True,
                    dropout_rate=0
                    ):
        super(ResUNetBaseline_S_DCLA_v1, self).__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                f_list=f_list,
                trilinear=trilinear,
                dropout_rate=dropout_rate
        )
        self.dcla = DCLA(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=3, fusion_kernel=1)
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
        
        x5 = self.dcla([x1, x2, x3, x4], x5)  # [B, 256, D/8, H/8, W/8]
        
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
