import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))


from nnArchitecture.commons import (
    init_weights_3d,
    UpSample
)

from nnArchitecture.nets.dcla_nets.dcla_unet_250603.modules import *
    
from utils.test_unet import test_unet

class ResUNetBaseline_S(nn.Module):
    __remark__ = """
    [Version]: 0602
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

# class ResUNeXt(nn.Module):
#     __remark__ = """
#     [Version]: 0603
#     [Author]: Junyin Xiong
#     [Features]
#     • 总参数量: 3.225M
#     • FLOPs: 195.083G
    
#     [测试集结果]

#     """
#     def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True):
#         super(ResUNeXt, self).__init__()
        
#         self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.Conv1 = ResNeXtConv(in_channels, f_list[0])
#         self.Conv2 = ResNeXtConv(f_list[0], f_list[1])
#         self.Conv3 = ResNeXtConv(f_list[1], f_list[2])
#         self.Conv4 = ResNeXtConv(f_list[2], f_list[3])
        
#         self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
#         self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
#         self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
#         self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
#         self.UpConv4 = ResNeXtConv(f_list[3]*2, f_list[3]//2)
#         self.UpConv3 = ResNeXtConv(f_list[2]*2, f_list[2]//2)
#         self.UpConv2 = ResNeXtConv(f_list[1]*2, f_list[1]//2)
#         self.UpConv1 = ResNeXtConv(f_list[0]*2, f_list[0])
        
#         self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
#         self.apply(init_weights_3d)  # 初始化权重
        
#     def forward(self, x):
#         # Encoder
#         x1 = self.Conv1(x)                # [B, 32, D, H, W]
#         x2 = self.MaxPool(x1)
#         x2 = self.Conv2(x2)      # [B, 64, D/2, H/2, W/2]
#         x3 = self.MaxPool(x2)
#         x3 = self.Conv3(x3)      # [B, 128, D/4, H/4, W/4]
#         x4 = self.MaxPool(x3)
#         x4 = self.Conv4(x4)      # [B, 256, D/8, H/8, W/8]
#         x5 = self.MaxPool(x4)
        
#         # Decoder with Attention
#         d5 = self.Up4(x5)               # [B, 256, D/8, H/8, W/8]
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.UpConv4(d5)    # [B, 128, D/8, H/8, W/8]
        
#         d4 = self.Up3(d5)        # [B, 128, D/4, H/4, W/4]
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.UpConv3(d4)    # [B, 64, D/4, H/4, W/4]
        
#         d3 = self.Up2(d4)        # [B, 64, D/2, H/2, W/2]
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.UpConv2(d3)    # [B, 32, D/2, H/2, W/2]
        
#         d2 = self.Up1(d3)        # [B, 32, D, H, W]
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.UpConv1(d2)    # [B, 32, D, H, W]
        
#         out = self.outc(d2)  # [B, out_channels, D, H, W]
#         return out

# class ResUNeXt_Attn(nn.Module):
#     __remark__ = """
#     [Version]: 0603
#     [Author]: Junyin Xiong
#     [Features]
#     • 总参数量: 1.147M
#     • FLOPs: 73.430G
    
#     [测试集结果]

#     """
#     def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True):
#         super(ResUNeXt_Attn, self).__init__()
        
#         self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.Conv1 = ResNeXtAttnConv(in_channels, f_list[0])
#         self.Conv2 = ResNeXtAttnConv(f_list[0], f_list[1])
#         self.Conv3 = ResNeXtAttnConv(f_list[1], f_list[2])
#         self.Conv4 = ResNeXtAttnConv(f_list[2], f_list[3])
        
#         self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
#         self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
#         self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
#         self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
#         self.UpConv4 = ResNeXtAttnConv(f_list[3]*2, f_list[3]//2)
#         self.UpConv3 = ResNeXtAttnConv(f_list[2]*2, f_list[2]//2)
#         self.UpConv2 = ResNeXtAttnConv(f_list[1]*2, f_list[1]//2)
#         self.UpConv1 = ResNeXtAttnConv(f_list[0]*2, f_list[0])
        
#         self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
#         self.apply(init_weights_3d)  # 初始化权重
        
#     def forward(self, x):
#         # Encoder
#         x1 = self.Conv1(x)                # [B, 32, D, H, W]
#         x2 = self.MaxPool(x1)
#         x2 = self.Conv2(x2)      # [B, 64, D/2, H/2, W/2]
#         x3 = self.MaxPool(x2)
#         x3 = self.Conv3(x3)      # [B, 128, D/4, H/4, W/4]
#         x4 = self.MaxPool(x3)
#         x4 = self.Conv4(x4)      # [B, 256, D/8, H/8, W/8]
#         x5 = self.MaxPool(x4)
        
#         # Decoder with Attention
#         d5 = self.Up4(x5)               # [B, 256, D/8, H/8, W/8]
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.UpConv4(d5)    # [B, 128, D/8, H/8, W/8]
        
#         d4 = self.Up3(d5)        # [B, 128, D/4, H/4, W/4]
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.UpConv3(d4)    # [B, 64, D/4, H/4, W/4]
        
#         d3 = self.Up2(d4)        # [B, 64, D/2, H/2, W/2]
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.UpConv2(d3)    # [B, 32, D/2, H/2, W/2]
        
#         d2 = self.Up1(d3)        # [B, 32, D, H, W]
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.UpConv1(d2)    # [B, 32, D, H, W]
        
#         out = self.outc(d2)  # [B, out_channels, D, H, W]
#         return out

class DCLA_UNet_250603(nn.Module):
    __remark__ = """
    [Version]: 250602
    [Author]: Junyin Xiong
    [Features]
    • 使用MSF做编码器
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
        super(DCLA_UNet_250603, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = MutilScaleFusionBlock(in_channels, f_list[0])
        self.Conv2 = MutilScaleFusionBlock(f_list[0], f_list[1])
        self.Conv3 = MutilScaleFusionBlock(f_list[1], f_list[2])
        self.Conv4 = MutilScaleFusionBlock(f_list[2], f_list[3])

        self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8)
        
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = MutilScaleFusionBlock(in_channels=f_list[3]*2, out_channels=f_list[3]//2)
        self.UpConv3 = MutilScaleFusionBlock(in_channels=f_list[2]*2, out_channels=f_list[2]//2)
        self.UpConv2 = MutilScaleFusionBlock(in_channels=f_list[1]*2, out_channels=f_list[1]//2)
        self.UpConv1 = MutilScaleFusionBlock(in_channels=f_list[0]*2, out_channels=f_list[0])
        
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
    
        x5 = self.dcla([x1, x2, x3, x4], x5) + x5 if hasattr(self, "dcla") else x5# [B, 256, D/8, H/8, W/8]  #加残差
        
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

class DCLA_UNet_250603_v2(nn.Module):
    __remark__ = """
    [Version]: 250602
    [Author]: Junyin Xiong
    [Features]
    • 使用MSF做编码器
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
        super(DCLA_UNet_250603_v2, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = DynamicScaleFusion(in_channels, f_list[0])
        self.Conv2 = DynamicScaleFusion(f_list[0], f_list[1])
        self.Conv3 = DynamicScaleFusion(f_list[1], f_list[2])
        self.Conv4 = DynamicScaleFusion(f_list[2], f_list[3])

        self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8)
        
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = DynamicScaleFusion(in_channels=f_list[3]*2, out_channels=f_list[3]//2)
        self.UpConv3 = DynamicScaleFusion(in_channels=f_list[2]*2, out_channels=f_list[2]//2)
        self.UpConv2 = DynamicScaleFusion(in_channels=f_list[1]*2, out_channels=f_list[1]//2)
        self.UpConv1 = DynamicScaleFusion(in_channels=f_list[0]*2, out_channels=f_list[0])
        
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
    
        x5 = self.dcla([x1, x2, x3, x4], x5) + x5 if hasattr(self, "dcla") else x5# [B, 256, D/8, H/8, W/8]  #加残差
        
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
    
class DCLA_UNet_250603_v3(nn.Module):
    __remark__ = """
    [Version]: 250602
    [Author]: Junyin Xiong
    [Features]
    • 使用MSF做编码器
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
        super(DCLA_UNet_250603_v3, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = DynamicMutilScaleFusionBlock(in_channels, f_list[0])
        self.Conv2 = DynamicMutilScaleFusionBlock(f_list[0], f_list[1])
        self.Conv3 = DynamicMutilScaleFusionBlock(f_list[1], f_list[2])
        self.Conv4 = DynamicMutilScaleFusionBlock(f_list[2], f_list[3])

        self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8)
        
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = DynamicMutilScaleFusionBlock(in_channels=f_list[3]*2, out_channels=f_list[3]//2)
        self.UpConv3 = DynamicMutilScaleFusionBlock(in_channels=f_list[2]*2, out_channels=f_list[2]//2)
        self.UpConv2 = DynamicMutilScaleFusionBlock(in_channels=f_list[1]*2, out_channels=f_list[1]//2)
        self.UpConv1 = DynamicMutilScaleFusionBlock(in_channels=f_list[0]*2, out_channels=f_list[0])
        
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
    
        x5 = self.dcla([x1, x2, x3, x4], x5) + x5 if hasattr(self, "dcla") else x5# [B, 256, D/8, H/8, W/8]  #加残差
        
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
    
class DCLA_UNet_250603_v4(nn.Module):
    __remark__ = """
    [Version]: 250602
    [Author]: Junyin Xiong
    [Features]
    • 使用MSF做编码器
    • 总参数量: 871.313K
    • FLOPs: 55.512G
    [Changes]
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
        super(DCLA_UNet_250603_v4, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = DynamicAxialConv3d(in_channels, f_list[0])
        self.Conv2 = DynamicAxialConv3d(f_list[0], f_list[1])
        self.Conv3 = DynamicAxialConv3d(f_list[1], f_list[2])
        self.Conv4 = DynamicAxialConv3d(f_list[2], f_list[3])

        self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8)
        
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = DynamicAxialConv3d(in_channels=f_list[3]*2, out_channels=f_list[3]//2)
        self.UpConv3 = DynamicAxialConv3d(in_channels=f_list[2]*2, out_channels=f_list[2]//2)
        self.UpConv2 = DynamicAxialConv3d(in_channels=f_list[1]*2, out_channels=f_list[1]//2)
        self.UpConv1 = DynamicAxialConv3d(in_channels=f_list[0]*2, out_channels=f_list[0])
        
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
    
        x5 = self.dcla([x1, x2, x3, x4], x5) + x5 if hasattr(self, "dcla") else x5# [B, 256, D/8, H/8, W/8]  #加残差
        
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
    
class BaseLine_S_DCLA_250603(DCLA_UNet_250603):
    __remark__ = """
    [Version]: 250602
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
        super(BaseLine_S_DCLA_250603, self).__init__(
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
    test_unet(model_class=DCLA_UNet_250603_v4, batch_size=1)   
    model = DCLA_UNet_250603_v4(in_channels=4, out_channels=4)
    print(model.__remark__)
    