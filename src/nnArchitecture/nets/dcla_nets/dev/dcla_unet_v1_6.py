# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/01 18:43
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建 DCLA UNet 训练模型
*      VERSION: v1.6
*      FEATURES: 
               - ✅构建 DCLA UNet v1 模型结构
               - ✅编码器使用SLK特征提取模块。Kernel_size=5
               - ✅使用 DCLA 进行跨层特征融合，并对编码器进行增强，下采样核为7
               - ✅解码器使用MSF多尺度融合模块
               - ✅DCLA 下采样核为3,5,7
*      STATE:     开发中
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from nnArchitecture.commons import (
    init_weights_3d,
    UpSample,
    DepthwiseAxialConv3d,
    LightweightChannelAttention3Dv2,
    LightweightSpatialAttention3D,
    LightweightChannelAttention3D
)

from nnArchitecture.nets.dcla_nets.dev.baseline_v1 import (
    ResUNetBaseline_S
)

from utils.test_unet import test_unet

class DCLA_UNet(nn.Module):
    __remark__ = """
    [Version]: V1.6
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK大核卷积 (kernel_size=7)
    • 集成DCLA跨层注意力机制(Down_kernel=7) 
    • DCLA输出引入残差结构 (V1.1)
    • SLK中引入通道MLP结构(V1.1)
    • DCLA 下采样核为3,5,7
    • 总参数量: 560.901K
    • FLOPs: 40.634G
    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True
                 ):
        super(DCLA_UNet, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = SlimLargeKernelBlock(in_channels, f_list[0], kernel_size=kernel_size)
        self.Conv2 = SlimLargeKernelBlock(f_list[0], f_list[1], kernel_size=kernel_size)
        self.Conv3 = SlimLargeKernelBlock(f_list[1], f_list[2], kernel_size=kernel_size)
        self.Conv4 = SlimLargeKernelBlock(f_list[2], f_list[3], kernel_size=kernel_size)
        
        self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=[3, 5, 7], fusion_kernel=1)
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
        self.kernel_size = 7
        self.Conv1 = SlimLargeKernelBlock(in_channels, f_list[0], kernel_size=self.kernel_size)
        self.Conv2 = SlimLargeKernelBlock(f_list[0], f_list[1], kernel_size=self.kernel_size)
        self.Conv3 = SlimLargeKernelBlock(f_list[1], f_list[2], kernel_size=self.kernel_size)
        self.Conv4 = SlimLargeKernelBlock(f_list[2], f_list[3], kernel_size=self.kernel_size)
        
    def forward(self, x):
        return super().forward(x)
    
    
class ResUNetBaseline_S_LiteMSF(ResUNetBaseline_S):
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
        self.Conv1 = SlimLargeKernelBlock(in_channels, f_list[0], kernel_size=7)  #编码器使用小核（3）的效果由于大核（7）
        self.Conv2 = SlimLargeKernelBlock(f_list[0], f_list[1], kernel_size=7)
        self.Conv3 = SlimLargeKernelBlock(f_list[1], f_list[2], kernel_size=7)
        self.Conv4 = SlimLargeKernelBlock(f_list[2], f_list[3], kernel_size=7)
        self.UpConv4 = MutilScaleFusionBlock(in_channels=f_list[3]*2, out_channels=f_list[3]//2,  fusion_kernel=7)
        self.UpConv3 = MutilScaleFusionBlock(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  fusion_kernel=7)
        self.UpConv2 = MutilScaleFusionBlock(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  fusion_kernel=7)
        self.UpConv1 = MutilScaleFusionBlock(in_channels=f_list[0]*2, out_channels=f_list[0],  fusion_kernel=7)
        
    def forward(self, x):
        return super().forward(x)


class ResUNetBaseline_S_DCLA(ResUNetBaseline_S):
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
    """
    2025-04-20 10:14:04,804  - CRITICAL - === [FINAL TEST METRIC] ===                                                      
    ╒═══════════════╤════════╤════════╤═══════╤═══════╕
    │ Metric_Name   │   MEAN │     ET │    TC │    WT │
    ╞═══════════════╪════════╪════════╪═══════╪═══════╡
    │ Dice          │  0.881 │  0.826 │ 0.889 │ 0.929 │
    ├───────────────┼────────┼────────┼───────┼───────┤
    │ Jaccard       │  0.816 │  0.748 │ 0.828 │ 0.873 │
    ├───────────────┼────────┼────────┼───────┼───────┤
    │ Precision     │  0.893 │  0.828 │ 0.91  │ 0.94  │
    ├───────────────┼────────┼────────┼───────┼───────┤
    │ Recall        │  0.888 │  0.844 │ 0.895 │ 0.925 │
    ├───────────────┼────────┼────────┼───────┼───────┤
    │ H95           │  6.388 │ 10.224 │ 5.15  │ 3.788 │
    ╘═══════════════╧════════╧════════╧═══════╧═══════╛
    Mean Loss: 0.1200;ET: 0.1758;ET: 0.1758;TC: 0.1114;WT: 0.0726
    
    """
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


""" =================================== Modules =================================== """

    
class SlimLargeKernelBlock(nn.Module): 
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
                    in_channels=out_channels, 
                    out_channels=out_channels//ratio, 
                    kernel_size=1
                    ),
                nn.GELU(),
                nn.Conv3d(
                    in_channels=out_channels//ratio, 
                    out_channels=out_channels, 
                    kernel_size=1
                    )
        )
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.GELU()
        
    def forward(self, x):
        out = self.depthwise(x)
        
        out += self.residual(x)
        return self.act(out)


class MutilScaleFusionBlock(nn.Module): #(MLP)
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
        # if mid_channels is None:
        #     mid_channels = out_channels*2
            
        # print(mid_channels==out_channels*2)
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
            self.ca = LightweightChannelAttention3D(out_channels, ratio=ratio)
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
    
class AdaptiveSpatialCondenser(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 kernel_size=[7,5,3], 
                 in_size=128, 
                 min_size=8,
                 fusion_mode='concat'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.in_size = in_size
        self.min_size = min_size
        self.fusion_mode = fusion_mode
        self.branches = self._build_multi_branchs()  # 动态构建下采样序列
        
    def _build_multi_branchs(self):
        return nn.ModuleList([self._build_single_branch(k) for k in self.kernel])
        
    def _build_single_branch(self, kernel_size):
        layers = []
        current_size = self.in_size
        
        # 动态构建下采样序列
        while current_size > self.min_size:
            layers.append(
                nn.Sequential(
                nn.Conv3d(
                    self.in_channels, 
                    self.out_channels, 
                    kernel_size=kernel_size, 
                    stride=2, 
                    padding=kernel_size//2
                    ),
                nn.BatchNorm3d(self.out_channels),
                nn.GELU()
            ))
            current_size = current_size // 2
        return nn.Sequential(*layers)
            
    def forward(self, x):
        branch_ouputs = [branch(x) for branch in self.branches]
        
        if self.fusion_mode == 'concat':
            return torch.cat(branch_ouputs, dim=1)
        elif self.fusion_mode == 'add':
            return torch.sum(torch.stack(branch_ouputs), dim=0)
        else:
            raise ValueError("Invalid fusion mode. Choose from 'concat' or 'add'.")

class DynamicCrossLevelAttention(nn.Module): #MSFA
    def __init__(self, 
                 ch_list, 
                 feats_size, 
                 min_size=8, 
                 squeeze_kernel=1,
                 down_kernel=[3,5,7], 
                 fusion_kernel=1,
                 fusion_mode='add'
                 ):
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
        self.kernel_size = down_kernel if isinstance(down_kernel, list) else [down_kernel]
        self.fusion_mode = fusion_mode
        self.squeeze_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        
        for ch in self.ch_list:
            self.squeeze_layers.append(
                nn.Sequential(
                    nn.Conv3d(ch, 1, kernel_size=squeeze_kernel, padding=squeeze_kernel//2),
                    nn.BatchNorm3d(1),
                    nn.GELU()
                    ))
        for feat_size in feats_size:
            self.down_layers.append(
                AdaptiveSpatialCondenser(
                    in_channels=1, 
                    out_channels=1, 
                    kernel_size=self.kernel_size, 
                    in_size=feat_size, 
                    min_size=8,
                    fusion_mode=self.fusion_mode  # 'concat' or 'add'
                )
            )
        self.conv = nn.Sequential(
            nn.Conv3d(len(self.kernel_size),
                      1, 
                      kernel_size=1, 
                      padding=0
                      ),
            nn.BatchNorm3d(1),
            nn.GELU(),
            nn.Conv3d(1, 1, kernel_size=1, padding=0)
        )
        self.fusion = nn.Conv3d(len(self.ch_list), 1, kernel_size=fusion_kernel, padding=fusion_kernel//2)

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
                down_feat = self.down_layers[i](feat)
            else:
                down_feat = feat
            if self.fusion_mode == 'concat':
                down_feat = self.conv(down_feat)
            elif self.fusion_mode == 'add':
                down_feat = down_feat
            else:
                raise ValueError("Invalid fusion mode. Choose from 'concat' or 'add'.")
            
            downs.append(down_feat.squeeze(1))
        # 特征融合
        fused = self.fusion(torch.stack(downs, dim=1))
        attn = torch.sigmoid(fused)
        
        out = attn * x + x
        return out
    

if __name__ == "__main__":
    test_unet(model_class=DCLA_UNet, batch_size=1)   
    model = DCLA_UNet(in_channels=4, out_channels=4)
    print(model.__remark__)