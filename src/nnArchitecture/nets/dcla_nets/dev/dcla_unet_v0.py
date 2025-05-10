# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/04/28 17:18:16
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建 DCLA UNet v0 结构，并测试DCLA的有效性
*      VERSION: v0.0
*      FEATURES: 
                - ✅构建 DCLA UNet v0 模型结构
                - ✅编码器使用SLK特征提取模块
                - ✅解码器使用MSFA特征融合模块
                - ✅DCLA 模块对编码器进行增强，下采样核为3
                
*      STATE:    结构搭建完成，测试通过
*      CHANGE ON: 
=================================================
'''
import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from nnArchitecture.commons import (
    init_weights_3d,
    UpSample,
    DepthwiseAxialConv3d,
    ResConv3D_S_BN,
    LightweightChannelAttention3D,
    LightweightSpatialAttention3D
)
from nnArchitecture.nets.dcla_nets.dev.baseline_v1 import (
    ResUNetBaseline_S
)

from utils.test_unet import test_unet

class DCLA_UNet_v0(nn.Module):
    __remark__ = """
    [Version]: V0.0
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK (Conv1x1x1 + DWAC7x7x7)
    • 集成DCLA跨层注意力机制(Down_Kernel=3)
    • 解码器采用MSF多尺度融合模块(融合层使用1x1)
    • 总参数量: 492.473K
    • FLOPs: 36.946G
    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True,
                 use_attn=True,
                 use_fusion=True
                 ):
        super(DCLA_UNet_v0, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = SlimLargeKernelBlock(in_channels, f_list[0], kernel_size=7)
        self.Conv2 = SlimLargeKernelBlock(f_list[0], f_list[1], kernel_size=7)
        self.Conv3 = SlimLargeKernelBlock(f_list[1], f_list[2], kernel_size=7)
        self.Conv4 = SlimLargeKernelBlock(f_list[2], f_list[3], kernel_size=7)
        
        self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=3, fusion_kernel=1)
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = MutilScaleFusionBlock(in_channels=f_list[3]*2, out_channels=f_list[3]//2,  use_attn=use_attn, use_fusion=use_fusion, fusion_kernel=7)
        self.UpConv3 = MutilScaleFusionBlock(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  use_attn=use_attn, use_fusion=use_fusion, fusion_kernel=7)
        self.UpConv2 = MutilScaleFusionBlock(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  use_attn=use_attn, use_fusion=use_fusion, fusion_kernel=7)
        self.UpConv1 = MutilScaleFusionBlock(in_channels=f_list[0]*2, out_channels=f_list[0],  use_attn=use_attn, use_fusion=use_fusion, fusion_kernel=7)
        
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

class ResUNetBaseline_S_SLK(ResUNetBaseline_S):
    __remark__ = """
    [Version]: V0.0
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK (Conv1x1x1 + DWAC7x7x7)
    • 总参数量: 967.268K
    • FLOPs: 91.196G
    """
    # 0.86 M
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0
                 ):
        super(ResUNetBaseline_S_SLK, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate
        )
        self.Conv1 = SlimLargeKernelBlock(in_channels, f_list[0], kernel_size=3)
        self.Conv2 = SlimLargeKernelBlock(f_list[0], f_list[1], kernel_size=3)
        self.Conv3 = SlimLargeKernelBlock(f_list[1], f_list[2], kernel_size=3)
        self.Conv4 = SlimLargeKernelBlock(f_list[2], f_list[3], kernel_size=3)
    
    def forward(self, x):
        return super().forward(x)
    

class ResUNetBaseline_S_SLK_LiteMSF(ResUNetBaseline_S_SLK):
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0
                 ):
        super(ResUNetBaseline_S_SLK_LiteMSF, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear,  
                dropout_rate=dropout_rate
        )
        self.Conv1 = SlimLargeKernelBlock(in_channels, f_list[0], kernel_size=5)  #编码器使用小核（3）的效果由于大核（7）
        self.Conv2 = SlimLargeKernelBlock(f_list[0], f_list[1], kernel_size=5)
        self.Conv3 = SlimLargeKernelBlock(f_list[1], f_list[2], kernel_size=5)
        self.Conv4 = SlimLargeKernelBlock(f_list[2], f_list[3], kernel_size=5)
        self.UpConv4 = MutilScaleFusionBlock(in_channels=f_list[3]*2, out_channels=f_list[3]//2)
        self.UpConv3 = MutilScaleFusionBlock(in_channels=f_list[2]*2, out_channels=f_list[2]//2)
        self.UpConv2 = MutilScaleFusionBlock(in_channels=f_list[1]*2, out_channels=f_list[1]//2)
        self.UpConv1 = MutilScaleFusionBlock(in_channels=f_list[0]*2, out_channels=f_list[0], mid_channels=f_list[0])
        
    def forward(self, x):
        return super().forward(x)
    
class ResUNetBaseline_S_DCLA(ResUNetBaseline_S):
    __remark__ = """
    [Version]: V0.0
    [Author]: Junyin Xiong
    [Features]
    • 使用DCLA模块
    • 总参数量: 3.225M
    • FLOPs: 195.183G
    """
    def __init__(self,
                    in_channels=4,
                    out_channels=4,
                    f_list=[32, 64, 128, 256],
                    trilinear=True,
                    dropout_rate=0
                    ):
        super(ResUNetBaseline_S_DCLA, self).__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                f_list=f_list,
                trilinear=trilinear,
                dropout_rate=dropout_rate
        )
        self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=3, fusion_kernel=1)
        
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

class ResUNetBaseline_S_LiteMSF(ResUNetBaseline_S):
    __remark__ = """
    [Version]: V0.0
    [Author]: Junyin Xiong
    [Features]
    • 解码器使用MSF多尺度融合模块
    • 总参数量: 2.718M
    • FLOPs: 137.933G
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0
                 ):
        super(ResUNetBaseline_S_LiteMSF, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate
        )
        
        self.UpConv4 = MutilScaleFusionBlock(in_channels=f_list[3]*2, out_channels=f_list[3]//2)
        self.UpConv3 = MutilScaleFusionBlock(in_channels=f_list[2]*2, out_channels=f_list[2]//2)
        self.UpConv2 = MutilScaleFusionBlock(in_channels=f_list[1]*2, out_channels=f_list[1]//2)
        self.UpConv1 = MutilScaleFusionBlock(in_channels=f_list[0]*2, out_channels=f_list[0])
        
    def forward(self, x):
        return super().forward(x)
    
    
class ResUNetBaseline_S_DCLA_SLK(ResUNetBaseline_S_DCLA):
    # 0.86 M
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0
                 ):
        super(ResUNetBaseline_S_DCLA_SLK, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate
        )
        self.Conv1 = SlimLargeKernelBlock(in_channels, f_list[0])
        self.Conv2 = SlimLargeKernelBlock(f_list[0], f_list[1])
        self.Conv3 = SlimLargeKernelBlock(f_list[1], f_list[2])
        self.Conv4 = SlimLargeKernelBlock(f_list[2], f_list[3])
          
    def forward(self, x):
        return super().forward(x)
    
class ResUNetBaseline_S_DCLA_LiteMSF(ResUNetBaseline_S_DCLA):
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0
                 ):
        super(ResUNetBaseline_S_DCLA_LiteMSF, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate
        )
        self.UpConv4 = MutilScaleFusionBlock(in_channels=f_list[3]*2, out_channels=f_list[3]//2)
        self.UpConv3 = MutilScaleFusionBlock(in_channels=f_list[2]*2, out_channels=f_list[2]//2)
        self.UpConv2 = MutilScaleFusionBlock(in_channels=f_list[1]*2, out_channels=f_list[1]//2)
        self.UpConv1 = MutilScaleFusionBlock(in_channels=f_list[0]*2, out_channels=f_list[0])
        
    def forward(self, x):
        return super().forward(x)

""" ======================================= Modules ======================================= """

class ResConv3D_S_BN(nn.Module):
    """(conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=0.2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=1),
            nn.Conv3d(out_channels, 
                      out_channels, 
                      kernel_size=kernel_size, 
                      stride=stride,
                      padding=padding
                      ),
            nn.BatchNorm3d(out_channels),
        )
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout3d(p=dropout_rate)  # 添加 Dropout 层
    def forward(self, x):
        out = self.double_conv(x) + self.residual(x)
        return self.relu(out)    

class DepthwiseAxialConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=None, groups= None):
        super().__init__()  # 首先调用父类的 __init__ 方法
        if dilation is not None:
            k_eq = (kernel_size - 1)*dilation + 1
            p_eq = (k_eq - 1)//2
            assert k_eq % 2 == 1, "kernel_size must be odd"
        
        self.AxialConv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(kernel_size,1, 1),
                stride=stride,
                padding=(p_eq,0, 0) if dilation is not None else (kernel_size//2,0, 0),
                groups=groups if groups is not None else in_channels,
                dilation=dilation if dilation is not None else 1
            ),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1,kernel_size, 1),
                stride=stride,
                padding=(0,p_eq, 0) if dilation is not None else (0,kernel_size//2, 0),
                groups=groups if groups is not None else in_channels,
                dilation=dilation if dilation is not None else 1
            ),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1,1, kernel_size),
                stride=stride,
                padding=(0,0, p_eq) if dilation is not None else (0,0, kernel_size//2),
                groups=groups if groups is not None else in_channels,
                dilation=dilation if dilation is not None else 1
            )
        )
        # 添加一个点卷积来改变通道数
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.AxialConv(x)
        x = self.pointwise(x)
        return x


# SlimLargeKernelBlock

class SlimLargeKernelBlock(nn.Module): 
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 groups=None,
                 ratio=2,
                 act_op='gelu'
                 ):
        super().__init__()
        self.depthwise =nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                DepthwiseAxialConv3d(
                    out_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size
                ),
                nn.BatchNorm3d(out_channels)
        )
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.ReLU()
        
    def forward(self, x):
        out = self.depthwise(x)
        
        out += self.residual(x)
        return self.act(out)
    
# MutilScaleFusionBlock
class MutilScaleFusionBlock(nn.Module): #(MLP)
    def __init__(self, 
                 in_channels, 
                 out_channels, 
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
        
        self.conv = nn.Conv3d(in_channels, out_channels, 1)
        # 创建多个具有不同核的分离卷积分支
        self.sep_branchs = nn.ModuleList([
            nn.Sequential(
                DepthwiseAxialConv3d(
                    out_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size, 
                    dilation=d
                ),
                nn.BatchNorm3d(out_channels),
                nn.GELU()
            ) for d in dilations
        ])
        if use_attn:
            self.ca = LightweightChannelAttention3D(out_channels, ratio=ratio)
            self.sa = LightweightSpatialAttention3D(kernel_size=7)
            
        if use_fusion: 
            self.fusion = nn.Sequential(
                nn.Conv3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1
                ),
                nn.BatchNorm3d(out_channels)
            ) 
        
        # 残差连接，如果输入输出通道数不同，使用1x1卷积调整；否则使用恒等映射
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # relu
        self.act = nn.ReLU() if act_op == 'relu' else nn.GELU()
        
    
    def forward(self, x):
        out = self.conv(x)
        sum_out = torch.sum(torch.stack([sep(out) for sep in self.sep_branchs]), dim=0)  # 使用加法而不是拼接
        if self.use_fusion:
            out = self.fusion(sum_out)
        
        if self.use_attn:
            channels_attn = self.ca(sum_out)
            spatial_attn = self.sa(sum_out)
            out = out * channels_attn * spatial_attn
        
        out += self.residual(x)
        return self.act(out)   

# DCLA

class AdaptiveSpatialCondenser(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=7, in_size=128, min_size=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.in_size = in_size
        self.min_size = min_size
        self.layers = self._build_layers()
        
        # 根据输入大小自动生成下采样层
    def _build_layers(self):
        layers = nn.ModuleList() 
        current_size = self.in_size
        while current_size > self.min_size:
            layers.append(
                nn.Sequential(
                nn.Conv3d(
                    self.in_channels, 
                    self.out_channels, 
                    kernel_size=self.kernel, 
                    stride=2, 
                    padding=self.kernel//2
                    ),
                nn.BatchNorm3d(self.out_channels),
                nn.ReLU()
            ))
            current_size = current_size // 2
        return layers
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DynamicCrossLevelAttention(nn.Module): #MSFA
    def __init__(self, ch_list, feats_size, min_size=8, squeeze_kernel=1, down_kernel=3, fusion_kernel=1):
        """
        Args: 
            ch_list: 输入特征的通道数
            feats_size: 输入特征的空间尺寸
            min_size: 最小空间尺寸
            kernel_size: 卷积核大小, 可以是一个整数或一个元组 (k1, k2, k3, k4)
        """
        super().__init__()
        self.ch_list = ch_list
        self.feats_size = feats_size
        self.min_size = min_size
        self.kernel_size = down_kernel
        self.squeeze_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        
        if isinstance(self.kernel_size, int):
            for ch in self.ch_list:
                self.squeeze_layers.append(
                    nn.Sequential(
                        nn.Conv3d(ch, 1, kernel_size=1),
                        nn.ReLU(inplace=True)
                        ))
            for feat_size in feats_size:
                self.down_layers.append(
                    AdaptiveSpatialCondenser(
                        in_channels=1, 
                        out_channels=1, 
                        kernel_size=self.kernel_size, 
                        in_size=feat_size, 
                        min_size=8
                    )
                )
        elif len(self.kernel_size) == 4:
            for k, ch in zip(self.kernel_size, self.ch_list):
                self.squeeze_layers.append(
                    nn.Sequential(
                        nn.Conv3d(ch, 1, kernel_size=k, padding=k//2, bias=False),
                        nn.ReLU(inplace=True)
                        ))
            for k, feat_size in zip(self.kernel_size, self.feats_size):
                self.down_layers.append(
                    AdaptiveSpatialCondenser(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=k,
                        in_size=feat_size,
                        min_size=8
                    )
                )
        else:
            raise ValueError("kernel_size must be an integer or a tuple of length 4.")
        
        self.fusion = nn.Conv3d(len(ch_list), 1, kernel_size=1)

    def forward(self, encoder_feats, x):
        squeezed_feats = []
        
        # 压缩通道数
        for i , squeeze_layer in enumerate(self.squeeze_layers):
            squeezed_feats.append(squeeze_layer(encoder_feats[i]))

        downs = []
        
        # 压缩空间维度
        for i, feat in enumerate(squeezed_feats):
            need_down = (feat.size(2) != self.min_size)
            if need_down:
                down_feat = self.down_layers[i](feat).squeeze(1)
            else:
                down_feat = feat.squeeze(1)
            downs.append(down_feat)
        # 特征融合
        fused = self.fusion(torch.stack(downs, dim=1))
        attn = torch.sigmoid(fused)
        
        out = attn * x
        return out



if __name__ == "__main__":
    test_unet(model_class=ResUNetBaseline_S_DCLA, batch_size=1)   
