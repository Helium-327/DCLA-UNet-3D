import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))


from nnArchitecture.commons import (
    init_weights_3d,
    UpSample
)

from nnArchitecture.nets.dcla_nets.dcla_unet_final.modules import *
    
from utils.test_unet import test_unet

class ResUNetBaseline_S(nn.Module):
    __remark__ = """
    [Version]: V2.0
    [Author]: Junyin Xiong
    [Features]
    • 总参数量: 3.225M
    • FLOPs: 195.083G
    
    [测试集结果]

    """
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True, dropout_rate=0):
        super(ResUNetBaseline_S, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = ResConv3D_S_BN(in_channels, f_list[0], dropout_rate=dropout_rate)
        self.Conv2 = ResConv3D_S_BN(f_list[0], f_list[1], dropout_rate=dropout_rate)
        self.Conv3 = ResConv3D_S_BN(f_list[1], f_list[2], dropout_rate=dropout_rate)
        self.Conv4 = ResConv3D_S_BN(f_list[2], f_list[3], dropout_rate=dropout_rate)
        
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        self.UpConv4 = ResConv3D_S_BN(f_list[3]*2, f_list[3]//2, dropout_rate=dropout_rate)
        self.UpConv3 = ResConv3D_S_BN(f_list[2]*2, f_list[2]//2, dropout_rate=dropout_rate)
        self.UpConv2 = ResConv3D_S_BN(f_list[1]*2, f_list[1]//2, dropout_rate=dropout_rate)
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

class DCLA_UNet_final(nn.Module):
    __remark__ = """
    [Version]: final version
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK大核卷积 (kernel_size=7)
    • 集成DCLA跨层注意力机制(Down_kernel=3, 5, 7) + 残差连接
    • 总参数量: 871.313K
    • FLOPs: 55.512G
    [Changes]
    • SLKv2
    • DCLAv1 
    [测试集结果]
    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True,
                 dropout_rate=0
                 ):
        super(DCLA_UNet_final, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = SlimLargeKernelBlock(in_channels, f_list[0], kernel_size=kernel_size)
        self.Conv2 = SlimLargeKernelBlock(f_list[0], f_list[1], kernel_size=kernel_size)
        self.Conv3 = SlimLargeKernelBlock(f_list[1], f_list[2], kernel_size=kernel_size)
        self.Conv4 = SlimLargeKernelBlock(f_list[2], f_list[3], kernel_size=kernel_size)

        self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=[7], fusion_kernel=1)
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = MutilScaleFusionBlock(in_channels=f_list[3]*2, out_channels=f_list[3]//2,  fusion_kernel=7)
        self.UpConv3 = MutilScaleFusionBlock(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  fusion_kernel=7)
        self.UpConv2 = MutilScaleFusionBlock(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  fusion_kernel=7)
        self.UpConv1 = MutilScaleFusionBlock(in_channels=f_list[0]*2, out_channels=f_list[0],     fusion_kernel=7)
        
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
    
        x5 = self.dcla([x1, x2, x3, x4], x5)  if hasattr(self,"dcla") else x5# [B, 256, D/8, H/8, W/8]
        
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
        
class DCLA_UNet_finalv2(nn.Module):
    __remark__ = """
    [Version]: final version
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK大核卷积 (kernel_size=7)
    • 集成DCLA跨层注意力机制(Down_kernel=3, 5, 7) + 残差连接
    • 总参数量: 871.313K
    • FLOPs: 55.512G
    [Changes]
    • SLKv2
    • DCLAv1 
    [测试集结果]
    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True,
                 dropout_rate=0
                 ):
        super(DCLA_UNet_finalv2, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = SlimLargeKernelBlock(in_channels, f_list[0], kernel_size=kernel_size)
        self.Conv2 = SlimLargeKernelBlock(f_list[0], f_list[1], kernel_size=kernel_size)
        self.Conv3 = SlimLargeKernelBlock(f_list[1], f_list[2], kernel_size=kernel_size)
        self.Conv4 = SlimLargeKernelBlock(f_list[2], f_list[3], kernel_size=kernel_size)

        self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=[7], fusion_kernel=1)
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = MutilScaleFusionBlock(in_channels=f_list[3]*2, out_channels=f_list[3]//2,  fusion_kernel=7)
        self.UpConv3 = MutilScaleFusionBlock(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  fusion_kernel=7)
        self.UpConv2 = MutilScaleFusionBlock(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  fusion_kernel=7)
        self.UpConv1 = MutilScaleFusionBlock(in_channels=f_list[0]*2, out_channels=f_list[0],     fusion_kernel=7)
        
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
    
        x5 = self.dcla([x1, x2, x3, x4], x5) + x5 if hasattr(self,"dcla") else x5# [B, 256, D/8, H/8, W/8]
        
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

class DCLA_UNet_finalv3(nn.Module):
    __remark__ = """
    [Version]: final version
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK大核卷积 (kernel_size=3)
    • 集成DCLA跨层注意力机制(Down_kernel=7) + 残差连接
    • 总参数量: 871.313K
    • FLOPs: 55.512G
    [Changes]
    • SLKv2
    • DCLAv1 
    [测试集结果]
    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 kernel_size=3, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True,
                 dropout_rate=0
                 ):
        super(DCLA_UNet_finalv3, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = SlimLargeKernelBlock(in_channels, f_list[0], kernel_size=kernel_size)
        self.Conv2 = SlimLargeKernelBlock(f_list[0], f_list[1], kernel_size=kernel_size)
        self.Conv3 = SlimLargeKernelBlock(f_list[1], f_list[2], kernel_size=kernel_size)
        self.Conv4 = SlimLargeKernelBlock(f_list[2], f_list[3], kernel_size=kernel_size)

        self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=[7], fusion_kernel=1)
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = MutilScaleFusionBlock(in_channels=f_list[3]*2, out_channels=f_list[3]//2,  fusion_kernel=7)
        self.UpConv3 = MutilScaleFusionBlock(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  fusion_kernel=7)
        self.UpConv2 = MutilScaleFusionBlock(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  fusion_kernel=7)
        self.UpConv1 = MutilScaleFusionBlock(in_channels=f_list[0]*2, out_channels=f_list[0],     fusion_kernel=7)
        
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
    
        x5 = self.dcla([x1, x2, x3, x4], x5) if hasattr(self,"dcla") else x5# [B, 256, D/8, H/8, W/8]
        
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

# TODO: 将每一层加DCLA 
class DCLA_UNet_finalv4(nn.Module):
    __remark__ = """
    [Version]: final version
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK大核卷积 (kernel_size=3)
    • 集成DCLA跨层注意力机制(Down_kernel=7) + 残差连接
    • 总参数量: 871.313K
    • FLOPs: 55.512G
    [Changes]
    • SLKv2
    • DCLAv1 
    [测试集结果]
    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 kernel_size=3, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True,
                 dropout_rate=0
                 ):
        super(DCLA_UNet_finalv4, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = SlimLargeKernelBlock(in_channels, f_list[0], kernel_size=kernel_size)
        self.Conv2 = SlimLargeKernelBlock(f_list[0], f_list[1], kernel_size=kernel_size)
        self.Conv3 = SlimLargeKernelBlock(f_list[1], f_list[2], kernel_size=kernel_size)
        self.Conv4 = SlimLargeKernelBlock(f_list[2], f_list[3], kernel_size=kernel_size)

        self.dcla1 = DynamicCrossLevelAttention(ch_list=f_list[:1], feats_size=[128], min_size=64, squeeze_kernel=1, down_kernel=[7], fusion_kernel=1)
        self.dcla2 = DynamicCrossLevelAttention(ch_list=f_list[1:2], feats_size=[64], min_size=32, squeeze_kernel=1, down_kernel=[7], fusion_kernel=1)
        self.dcla3 = DynamicCrossLevelAttention(ch_list=f_list[2:3], feats_size=[32], min_size=16, squeeze_kernel=1, down_kernel=[7], fusion_kernel=1)
        self.dcla4 = DynamicCrossLevelAttention(ch_list=[f_list[-1]], feats_size=[16], min_size=8, squeeze_kernel=1, down_kernel=[7], fusion_kernel=1)
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = MutilScaleFusionBlock(in_channels=f_list[3]*2, out_channels=f_list[3]//2,  fusion_kernel=7)
        self.UpConv3 = MutilScaleFusionBlock(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  fusion_kernel=7)
        self.UpConv2 = MutilScaleFusionBlock(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  fusion_kernel=7)
        self.UpConv1 = MutilScaleFusionBlock(in_channels=f_list[0]*2, out_channels=f_list[0],     fusion_kernel=7)
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
        self.apply(init_weights_3d)  # 初始化权重
        
    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)                # [B, 32, D, H, W]
        x2 = self.MaxPool(x1)
        x2 = self.dcla1([x1], x2) if hasattr(self, "dcla1") else x2 # [B, 32, D/2, H/2, W/2]
        
        x2 = self.Conv2(x2)      # [B, 64, D/2, H/2, W/2]
        x3 = self.MaxPool(x2)
        x3 = self.dcla2([x2], x3) if hasattr(self, "dcla2") else x3 # [B, 32, D/2, H/2, W/2]
        
        x3 = self.Conv3(x3)      # [B, 128, D/4, H/4, W/4]
        x4 = self.MaxPool(x3)
        x4 = self.dcla3([x3], x4) if hasattr(self, "dcla3") else x4
        
        x4 = self.Conv4(x4)      # [B, 256, D/8, H/8, W/8]
        x5 = self.MaxPool(x4)
        x5 = self.dcla4([x4], x5) if hasattr(self,"dcla4") else x5 # [B, 256, D/8, H/8, W/8]
        
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

class BaseLine_S_SLK_final(DCLA_UNet_final):
    __remark__ = """
    [Version]: final version
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 1.023M
    • FLOPs: 95.042G
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0
                 ):
        super(BaseLine_S_SLK_final, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate
        )
        if hasattr(self, 'dcla'):
            delattr(self, 'dcla')
            
        self.UpConv4 = ResConv3D_S_BN(f_list[3]*2, f_list[3]//2, dropout_rate=dropout_rate)
        self.UpConv3 = ResConv3D_S_BN(f_list[2]*2, f_list[2]//2, dropout_rate=dropout_rate)
        self.UpConv2 = ResConv3D_S_BN(f_list[1]*2, f_list[1]//2, dropout_rate=dropout_rate)
        self.UpConv1 = ResConv3D_S_BN(f_list[0]*2, f_list[0], dropout_rate=dropout_rate)      
        
    def forward(self, x):
        return super(BaseLine_S_SLK_final, self).forward(x) 
    
class BaseLine_S_DCLA_SLK_final(DCLA_UNet_final):
    __remark__ = """
    [Version]: V2
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 1.027M
    • FLOPs: 94.544G
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0
                 ):
        super(BaseLine_S_DCLA_SLK_final, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate
        )

        self.UpConv4 = ResConv3D_S_BN(f_list[3]*2, f_list[3]//2, dropout_rate=dropout_rate)
        self.UpConv3 = ResConv3D_S_BN(f_list[2]*2, f_list[2]//2, dropout_rate=dropout_rate)
        self.UpConv2 = ResConv3D_S_BN(f_list[1]*2, f_list[1]//2, dropout_rate=dropout_rate)
        self.UpConv1 = ResConv3D_S_BN(f_list[0]*2, f_list[0], dropout_rate=dropout_rate)      
        
    def forward(self, x):
        return super(BaseLine_S_DCLA_SLK_final, self).forward(x)

class BaseLine_S_MSF_final(DCLA_UNet_final):
    __remark__ = """
    [Version]: final version
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 2.877M
    • FLOPs: 146.865G
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0
                 ):
        super(BaseLine_S_MSF_final, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate
        )
        if hasattr(self, 'dcla'):
            delattr(self, 'dcla')
            
        self.Conv1 = ResConv3D_S_BN(in_channels, f_list[0], dropout_rate=dropout_rate)
        self.Conv2 = ResConv3D_S_BN(f_list[0], f_list[1], dropout_rate=dropout_rate)
        self.Conv3 = ResConv3D_S_BN(f_list[1], f_list[2], dropout_rate=dropout_rate)
        self.Conv4 = ResConv3D_S_BN(f_list[2], f_list[3], dropout_rate=dropout_rate)
        
    def forward(self, x):
        return super().forward(x) 
    
class BaseLine_S_DCLA_MSF_final(DCLA_UNet_final):
    __remark__ = """
    [Version]: final version
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 2.881M
    • FLOPs: 147.082G
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0
                 ):
        super(BaseLine_S_DCLA_MSF_final, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate
        )
        self.Conv1 = ResConv3D_S_BN(in_channels, f_list[0], dropout_rate=dropout_rate)
        self.Conv2 = ResConv3D_S_BN(f_list[0], f_list[1], dropout_rate=dropout_rate)
        self.Conv3 = ResConv3D_S_BN(f_list[1], f_list[2], dropout_rate=dropout_rate)
        self.Conv4 = ResConv3D_S_BN(f_list[2], f_list[3], dropout_rate=dropout_rate)
        
    def forward(self, x):
        return super().forward(x)
    
    
class BaseLine_S_SLK_MSF_final(DCLA_UNet_final):
    __remark__ = """
    [Version]: V2
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 675.436K
    • FLOPs: 46.108G
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(BaseLine_S_SLK_MSF_final, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear
        )
        if hasattr(self, 'dcla'):
            delattr(self, 'dcla')
        
    def forward(self, x):
        return super().forward(x) 
    
class BaseLine_S_DCLA_final(DCLA_UNet_final):
    __remark__ = """
    [Version]: final version
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 3.375M
    • FLOPs: 202.029G
    """
    def __init__(self,
                    in_channels=4,
                    out_channels=4,
                    f_list=[32, 64, 128, 256],
                    trilinear=True,
                    dropout_rate=0
                    ):
        super(BaseLine_S_DCLA_final, self).__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                f_list=f_list,
                trilinear=trilinear,
                dropout_rate=dropout_rate
        )
        self.Conv1 = ResConv3D_S_BN(in_channels, f_list[0], dropout_rate=dropout_rate)
        self.Conv2 = ResConv3D_S_BN(f_list[0], f_list[1], dropout_rate=dropout_rate)
        self.Conv3 = ResConv3D_S_BN(f_list[1], f_list[2], dropout_rate=dropout_rate)
        self.Conv4 = ResConv3D_S_BN(f_list[2], f_list[3], dropout_rate=dropout_rate)
        
        self.UpConv4 = ResConv3D_S_BN(f_list[3]*2, f_list[3]//2, dropout_rate=dropout_rate)
        self.UpConv3 = ResConv3D_S_BN(f_list[2]*2, f_list[2]//2, dropout_rate=dropout_rate)
        self.UpConv2 = ResConv3D_S_BN(f_list[1]*2, f_list[1]//2, dropout_rate=dropout_rate)
        self.UpConv1 = ResConv3D_S_BN(f_list[0]*2, f_list[0], dropout_rate=dropout_rate)
        
    def forward(self, x):
        return super().forward(x)      
    
if __name__ == "__main__":
    test_unet(model_class=DCLA_UNet_finalv4, batch_size=1)   
    model = DCLA_UNet_finalv4(in_channels=4, out_channels=4)
    print(model.__remark__)
    