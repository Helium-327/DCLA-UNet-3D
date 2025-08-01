'''
================================================
*      CREATE ON: 2025/04/30 17:23:32
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建 DCLA UNet v2 ，并测试v2结构的有效性
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

from nnArchitecture.nets.dcla_nets.dcla_unet_final.modules import *

from utils.test_unet import test_unet

""" ====================================== Baseline ====================================== """
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

# class DCLA_UNet_final(nn.Module):
#     __remark__ = """
#     [Version]: _final
#     [Author]: Junyin Xiong
#     [Features]
#     • 使用MSF做编码器
#     • 总参数量: 863.065K
#     • FLOPs: 64.566G
#     [Changes]
#     [测试集结果]
#     """
#     def __init__(self, 
#                  in_channels=4, 
#                  out_channels=4,
#                  kernel_size=7, 
#                  f_list=[32, 64, 128, 256], 
#                  trilinear=True,
#                  se_ratio=16,
#                  dropout_rate=0
#                  ):
#         super(DCLA_UNet_final, self).__init__()
#         self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.Conv1 = self._make_encoder_layer(in_channels, f_list[0], kernel_size=kernel_size, se_ratio=se_ratio)
#         self.Conv2 = self._make_encoder_layer(f_list[0], f_list[1], kernel_size=kernel_size, se_ratio=se_ratio)
#         self.Conv3 = self._make_encoder_layer(f_list[1], f_list[2], kernel_size=kernel_size, se_ratio=se_ratio)
#         self.Conv4 = self._make_encoder_layer(f_list[2], f_list[3], kernel_size=kernel_size, se_ratio=se_ratio)

#         self.dcla = DynamicCrossLevelAttention(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=[7], fusion_kernel=1)
#         self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
#         self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
#         self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
#         self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
#         self.UpConv4 = self._make_decoder_layer(f_list[3]*2, f_list[3]//2)
#         self.UpConv3 = self._make_decoder_layer(f_list[2]*2, f_list[2]//2)
#         self.UpConv2 = self._make_decoder_layer(f_list[1]*2, f_list[1]//2)
#         self.UpConv1 = self._make_decoder_layer(f_list[0]*2, f_list[0])
        
#         self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
#         self.apply(init_weights_3d)  # 初始化权重
        
#     def _make_encoder_layer(self, in_channels, out_channels, kernel_size, se_ratio=16):
#         return SlimLargeKernelBlockv4(in_channels, out_channels, kernel_size=kernel_size)
        
#     def _make_decoder_layer(self, 
#                       in_channels, 
#                       out_channels, 
#                       se_ratio=8,
#                       norm_type="batch", 
#                       act_type="gelu"
#                       ):
#         return nn.Sequential(
#             MutilScaleFusionBlock(in_channels, out_channels, dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type),
#             EfficientAttentionBlock(out_channels, ratio=16, spatial_kernal=7, parallel=True),
#         )

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
    
#         x5 = self.dcla([x1, x2, x3, x4], x5) + x5  if hasattr(self, "dcla") else x5      # [B, 256, D/8, H/8, W/8]
        
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
 
class DCLA_UNet_final(nn.Module):
    __remark__ = """
    [Version]: _final
    [Author]: Junyin Xiong
    [Features]
    • 使用MSF做编码器
    • 总参数量: 526.781K
    • FLOPs: 39.232G
    [Changes]
    [测试集结果]
    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256], 
                 trilinear=True,
                 se_ratio=16,
                 dropout_rate=0
                 ):
        super(DCLA_UNet_final, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = self._make_encoder_layer(in_channels, f_list[0], kernel_size=kernel_size, se_ratio=se_ratio)
        self.Conv2 = self._make_encoder_layer(f_list[0], f_list[1], kernel_size=kernel_size, se_ratio=se_ratio)
        self.Conv3 = self._make_encoder_layer(f_list[1], f_list[2], kernel_size=kernel_size, se_ratio=se_ratio)
        self.Conv4 = self._make_encoder_layer(f_list[2], f_list[3], kernel_size=kernel_size, se_ratio=se_ratio)

        self.dcla = DynamicCrossLevelAttentionv2(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=[3,5], fusion_kernel=3)
        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = self._make_decoder_layer(f_list[3]*2, f_list[3]//2)
        self.UpConv3 = self._make_decoder_layer(f_list[2]*2, f_list[2]//2)
        self.UpConv2 = self._make_decoder_layer(f_list[1]*2, f_list[1]//2)
        self.UpConv1 = self._make_decoder_layer(f_list[0]*2, f_list[0])
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
        self.apply(init_weights_3d)  # 初始化权重
        
    def _make_encoder_layer(self, in_channels, out_channels, kernel_size, se_ratio=16):
        return nn.Sequential(
            SlimLargeKernelBlockv4(in_channels, out_channels, kernel_size=kernel_size)
        )
        
    def _make_decoder_layer(self, 
                      in_channels, 
                      out_channels, 
                      se_ratio=8,
                      norm_type="batch", 
                      act_type="gelu"
                      ):
        return nn.Sequential(
            MutilScaleFusionBlock(in_channels, out_channels, dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
        )

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
    
        x5 = self.dcla([x1, x2, x3, x4], x5)  if hasattr(self, "dcla") else x5      # [B, 256, D/8, H/8, W/8]
        
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
    
class DCLA_UNet_DCLAv2_NoRes_final(DCLA_UNet_final):
    __remark__ = """
    [Version]: _final
    [Author]: Junyin Xiong
    [Features]
    • 集成DCLA跨层注意力机制(Down_kernel=3, 5, 7) + 残差连接
    • 总参数量: 2.852M
    • FLOPs: 145.133G
    [Changes]
    • 添加了ResNeXt模块
    """
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256],
                 dropout_rate=0.2,
                 norm_type="batch",
                 act_type="gelu",
                 ):
        super(DCLA_UNet_DCLAv2_NoRes_final, self).__init__()
        
        self.dcla = DynamicCrossLevelAttentionv2(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=[3,5], fusion_kernel=3)
    def forward(self, x):
        return super().forward(x)
 
class SLK_UNet_final(DCLA_UNet_final):
    __remark__ = """
    [Version]: _final
    [Author]: Junyin Xiong
    [Features]
    • 集成DCLA跨层注意力机制(Down_kernel=3, 5, 7) + 残差连接
    • 总参数量: 2.852M
    • FLOPs: 145.133G
    [Changes]
    • 添加了ResNeXt模块
    """
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256],
                 dropout_rate=0.2,
                 norm_type="batch",
                 act_type="gelu",
                 ):
        super(SLK_UNet_final, self).__init__()
        if hasattr(self, "dcla"):
            delattr(self, "dcla")
        
        self.Conv1 = SlimLargeKernelBlockv4(in_channels, f_list[0], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.Conv2 = SlimLargeKernelBlockv4(f_list[0], f_list[1], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.Conv3 = SlimLargeKernelBlockv4(f_list[1], f_list[2], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.Conv4 = SlimLargeKernelBlockv4(f_list[2], f_list[3], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        
        self.UpConv4 = SlimLargeKernelBlockv4(f_list[3]*2, f_list[3]//2, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.UpConv3 = SlimLargeKernelBlockv4(f_list[2]*2, f_list[2]//2, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.UpConv2 = SlimLargeKernelBlockv4(f_list[1]*2, f_list[1]//2, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.UpConv1 = SlimLargeKernelBlockv4(f_list[0]*2, f_list[0], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
    def forward(self, x):
        return super().forward(x)

class MSF_UNet_final(DCLA_UNet_final):
    __remark__ = """
    [Version]: _final
    [Author]: Junyin Xiong
    [Features]
    • 集成DCLA跨层注意力机制(Down_kernel=3, 5, 7) + 残差连接
    • 总参数量: 2.852M
    • FLOPs: 145.133G
    [Changes]
    • 添加了ResNeXt模块
    """
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256],
                 dropout_rate=0.2,
                 norm_type="batch",
                 act_type="gelu",
                 ):
        super(MSF_UNet_final, self).__init__()
        if hasattr(self, "dcla"):
            delattr(self, "dcla")
        self.Conv1 = MutilScaleFusionBlock(in_channels, f_list[0], dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
        self.Conv2 = MutilScaleFusionBlock(f_list[0], f_list[1], dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
        self.Conv3 = MutilScaleFusionBlock(f_list[1], f_list[2], dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
        self.Conv4 = MutilScaleFusionBlock(f_list[2], f_list[3], dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
        
        self.UpConv4 = MutilScaleFusionBlock(f_list[3]*2, f_list[3]//2, dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
        self.UpConv3 = MutilScaleFusionBlock(f_list[2]*2, f_list[2]//2, dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
        self.UpConv2 = MutilScaleFusionBlock(f_list[1]*2, f_list[1]//2, dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
        self.UpConv1 = MutilScaleFusionBlock(f_list[0]*2, f_list[0], dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
    def forward(self, x):
        return super().forward(x)

class SLK_MSF_UNet_final(DCLA_UNet_final):
    __remark__ = """
    [Version]: _final
    [Author]: Junyin Xiong
    [Features]
    • 集成DCLA跨层注意力机制(Down_kernel=3, 5, 7) + 残差连接
    • 总参数量: 2.852M
    • FLOPs: 145.133G
    [Changes]
    • 添加了ResNeXt模块
    """
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size=7, 
                 f_list=[32, 64, 128, 256],
                 dropout_rate=0.2,
                 norm_type="batch",
                 act_type="gelu",
                 ):
        super(SLK_MSF_UNet_final, self).__init__()
        if hasattr(self, "dcla"):
            delattr(self, "dcla")
        
        self.Conv1 = SlimLargeKernelBlockv4(in_channels, f_list[0], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.Conv2 = SlimLargeKernelBlockv4(f_list[0], f_list[1], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.Conv3 = SlimLargeKernelBlockv4(f_list[1], f_list[2], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.Conv4 = SlimLargeKernelBlockv4(f_list[2], f_list[3], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        
        self.UpConv4 = MutilScaleFusionBlock(f_list[3]*2, f_list[3]//2, dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
        self.UpConv3 = MutilScaleFusionBlock(f_list[2]*2, f_list[2]//2, dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
        self.UpConv2 = MutilScaleFusionBlock(f_list[1]*2, f_list[1]//2, dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
        self.UpConv1 = MutilScaleFusionBlock(f_list[0]*2, f_list[0], dilations=[1, 2, 3], norm_type=norm_type, act_type=act_type)
    def forward(self, x):
        return super().forward(x)

if __name__ == "__main__":
    test_unet(model_class=DCLA_UNet_final, batch_size=1)   
    model = DCLA_UNet_final(in_channels=4, out_channels=4)
    print(model.__remark__)
    