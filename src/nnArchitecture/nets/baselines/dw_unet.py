# -*- coding: UTF-8 -*-
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from nnArchitecture.commons import (
    init_weights_3d, 
    UpSample,
    DepthwiseAxialConv3d,
    DWResConv3D,
    LightweightChannelAttention3Dv2,
    LightweightSpatialAttention3D
)
from utils.test_unet import test_unet

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
class MutilScaleFusionBlock(nn.Module): #(MLP)
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 mid_channels=None,
                 kernel_size=3, 
                 dilations=[1,2,3], 
                 fusion_kernel=3,  # 尺度和编码器保持一致
                 ratio=2,               # 4比2好
                 act_op='gelu',
                 use_attn=False,
                 use_fusion=True
                 ):
        """Multi kernel separable convolution fusion block v4
        
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
        if mid_channels is None:
            mid_channels = out_channels*2
        # 创建多个具有不同核的分离卷积分支
        self.sep_branchs = nn.ModuleList([
            nn.Sequential(
                DepthwiseAxialConv3d(
                    in_channels,
                    mid_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size, 
                    dilation=d
                ),
                nn.BatchNorm3d(mid_channels),
                nn.GELU()
            ) for d in dilations
        ])
        if use_attn:
            self.ca = LightweightChannelAttention3Dv2(mid_channels, ratio=ratio)
            self.sa = LightweightSpatialAttention3D(kernel_size=7)
            
        if use_fusion: 
            self.fusion = nn.Sequential(
                nn.Conv3d(
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=1
                ),
                nn.BatchNorm3d(out_channels)
            ) 
        
        # 残差连接，如果输入输出通道数不同，使用1x1卷积调整；否则使用恒等映射
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # relu
        self.act = nn.ReLU() if act_op == 'relu' else nn.GELU()
        
        # self.bn = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        out = torch.sum(torch.stack([sep(x) for sep in self.sep_branchs]), dim=0)  # 使用加法而不是拼接
        
        if self.use_attn:
            channels_attn = self.ca(out)
            spatial_attn = self.sa(out)
            out = out * channels_attn * spatial_attn
            
        if self.use_fusion:
            out = self.fusion(out)
            
        out += self.residual(x)
        return self.act(out)   
    
class DWConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise =nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            dilation=dilation
            )
        self.pointwise = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class DWResConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, dropout_rate=0, act_op='relu'):
        super().__init__()
        self.dwconv = DWConv3D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
        self.residual = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels!= out_channels else nn.Identity()
        self.drop = nn.Dropout3d(p=dropout_rate)
        
        if act_op == 'gelu':
            self.act = nn.GELU()
        elif act_op == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError('act_op must be one of [gelu, relu]')

    def forward(self, x):
        out = self.dwconv(x)
        out += self.residual(x)
        out = self.drop(out)
        return self.act(out)
    
    
if __name__ == "__main__":
    test_unet(model_class=DWResUNet, batch_size=1)   