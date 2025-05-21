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

from nnArchitecture.nets.dcla_nets.dcla_unet_v3.mm import (
    ResConv3D_S_BN,
    # SlimLargeKernelBlock as SLK,
    # SlimLargeKernelBlockv1 as SLKv1,
    SlimLargeKernelBlockv2 as SLKv2,
    # SlimLargeKernelBlockv3 as SLKv3,
    MutilScaleFusionBlock as MSF,
    DynamicCrossLevelAttention as DCLA,
    DynamicCrossLevelAttentionv1 as DCLAv1,
    # DynamicCrossLevelAttentionv2 as DCLAv2,
    # AttentionGate ,
    # ChannelsAttentionBlock
)
    
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


""" ====================================== v2.0 ====================================== """


# class DCLA_UNet_v3(nn.Module):
#     __remark__ = """
#     [Version]: V2.4
#     [Author]: Junyin Xiong
#     [Features]
#     • 编码器使用SLK大核卷积 (kernel_size=7)
#     • 集成DCLA跨层注意力机制(Down_kernel=3, 5, 7) + 残差连接
#     • 总参数量: 865.689K
#     • FLOPs: 54.735G
#     [Changes]
#     • SLKv2
#     • DCLAv1 
#     [测试集结果]
#     """
#     def __init__(self, 
#                  in_channels=4, 
#                  out_channels=4,
#                  kernel_size=7, 
#                  f_list=[32, 64, 128, 256], 
#                  trilinear=True,
#                  dropout_rate=0,
#                  norm_type = 'batch',
#                  act_type = 'relu',
#                  ):
#         super(DCLA_UNet_v3, self).__init__()
#         self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.Conv1 = SLKv2(in_channels, f_list[0], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
#         self.dcla1 = DCLA(ch_list=[f_list[0]], feats_size=[128], min_size=64, squeeze_kernel=1, down_kernel=[3, 5, 7], fusion_kernel=1, norm_type=norm_type, act_type=act_type)
#         self.Conv2 = SLKv2(f_list[0],f_list[1], kernel_size=kernel_size,norm_type=norm_type, act_type=act_type)
#         self.dcla2 = DCLA(ch_list=f_list[:1], feats_size=[128, 64],  min_size=32, squeeze_kernel=1, down_kernel=[3, 5, 7], fusion_kernel=1, norm_type=norm_type, act_type=act_type)
#         self.Conv3 = SLKv2(f_list[1], f_list[2], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
#         self.dcla3 = DCLA(ch_list=f_list[:2], feats_size=[128, 64, 32], min_size=16, squeeze_kernel=1, down_kernel=[3, 5, 7], fusion_kernel=1, norm_type=norm_type, act_type=act_type)
#         self.Conv4 = SLKv2(f_list[2], f_list[3], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
#         self.dcla4 = DCLA(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=[3, 5, 7], fusion_kernel=1, norm_type=norm_type, act_type=act_type)

#         self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
#         self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
#         self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
#         self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
#         self.UpConv4 = MSF(in_channels=f_list[3]*2, out_channels=f_list[3]//2,  fusion_kernel=7, norm_type=norm_type, act_type=act_type)
#         self.UpConv3 = MSF(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  fusion_kernel=7, norm_type=norm_type, act_type=act_type)
#         self.UpConv2 = MSF(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  fusion_kernel=7, norm_type=norm_type, act_type=act_type)
#         self.UpConv1 = MSF(in_channels=f_list[0]*2, out_channels=f_list[0],     fusion_kernel=7, norm_type=norm_type, act_type=act_type)
        
#         self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
#         self.apply(init_weights_3d)  # 初始化权重
        
#     def forward(self, x):
#         # Encoder
#         x1 = self.Conv1(x)                # [B, 32, D, H, W]
#         x2 = self.MaxPool(x1)
#         x2 = self.dcla1([x1], x2) if hasattr(self, 'dcla1') else x2
        
#         x2 = self.Conv2(x2)      # [B, 64, D/2, H/2, W/2]
#         x3 = self.MaxPool(x2)
#         x3 = self.dcla2([x1, x2], x3) if hasattr(self, 'dcla2') else x3
        
        
#         x3 = self.Conv3(x3)      # [B, 128, D/4, H/4, W/4]
#         x4 = self.MaxPool(x3)
#         x4 = self.dcla3([x1, x2, x3], x4) if hasattr(self, 'dcla3') else x4
        
#         x4 = self.Conv4(x4)      # [B, 256, D/8, H/8, W/8]
#         x5 = self.MaxPool(x4)
#         x5 = self.dcla4([x1, x2, x3, x4], x5) if hasattr(self, 'dcla4') else x5
        
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


class DCLA_UNet_v3(nn.Module):
    __remark__ = """
    [Version]: V2.4
    [Author]: Junyin Xiong
    [Features]
    • 编码器使用SLK大核卷积 (kernel_size=7)
    • 集成DCLA跨层注意力机制(Down_kernel=3, 5, 7) + 残差连接
    • 总参数量: 865.689K
    • FLOPs: 54.735G
    [Changes]
    • SLKv2
    • DCLAv1 
    [测试集结果]
    """
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 kernel_size=7, 
                 down_kernels = [7],
                 f_list=[32, 64, 128, 256], 
                 trilinear=True,
                 dropout_rate=0,
                 norm_type = 'batch',
                 act_type = 'relu',
                 ):
        super(DCLA_UNet_v3, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = SLKv2(in_channels, f_list[0], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.dcla1 = DCLA(ch_list=f_list[:1], feats_size=[128], min_size=64, squeeze_kernel=1, down_kernel=down_kernels, fusion_kernel=1, norm_type=norm_type, act_type=act_type)
        self.Conv2 = SLKv2(f_list[0],f_list[1], kernel_size=kernel_size,norm_type=norm_type, act_type=act_type)
        self.dcla2 = DCLA(ch_list=f_list[:2], feats_size=[128, 64],  min_size=32, squeeze_kernel=1, down_kernel=down_kernels, fusion_kernel=1, norm_type=norm_type, act_type=act_type)
        self.Conv3 = SLKv2(f_list[1], f_list[2], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.dcla3 = DCLA(ch_list=f_list[:3], feats_size=[128, 64, 32], min_size=16, squeeze_kernel=1, down_kernel=down_kernels, fusion_kernel=1, norm_type=norm_type, act_type=act_type)
        self.Conv4 = SLKv2(f_list[2], f_list[3], kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.dcla4 = DCLA(ch_list=f_list, feats_size=[128, 64, 32, 16], min_size=8, squeeze_kernel=1, down_kernel=down_kernels, fusion_kernel=1, norm_type=norm_type, act_type=act_type)

        self.Up4 = UpSample(f_list[3], f_list[3], trilinear)
        self.Up3 = UpSample(f_list[2], f_list[2], trilinear)
        self.Up2 = UpSample(f_list[1], f_list[1], trilinear)
        self.Up1 = UpSample(f_list[0], f_list[0], trilinear)
        
        self.UpConv4 = MSF(in_channels=f_list[3]*2, out_channels=f_list[3]//2,  fusion_kernel=7, norm_type=norm_type, act_type=act_type)
        self.UpConv3 = MSF(in_channels=f_list[2]*2, out_channels=f_list[2]//2,  fusion_kernel=7, norm_type=norm_type, act_type=act_type)
        self.UpConv2 = MSF(in_channels=f_list[1]*2, out_channels=f_list[1]//2,  fusion_kernel=7, norm_type=norm_type, act_type=act_type)
        self.UpConv1 = MSF(in_channels=f_list[0]*2, out_channels=f_list[0],     fusion_kernel=7, norm_type=norm_type, act_type=act_type)
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
        self.apply(init_weights_3d)  # 初始化权重
        
    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)                # [B, 32, D, H, W]
        x2 = self.MaxPool(x1)
        x2 = self.dcla1([x1], x2) if hasattr(self, 'dcla1') else x2
        
        x2 = self.Conv2(x2)      # [B, 64, D/2, H/2, W/2]
        x3 = self.MaxPool(x2)
        x3 = self.dcla2([x1, x2], x3) if hasattr(self, 'dcla2') else x3
        
        
        x3 = self.Conv3(x3)      # [B, 128, D/4, H/4, W/4]
        x4 = self.MaxPool(x3)
        x4 = self.dcla3([x1, x2, x3], x4) if hasattr(self, 'dcla3') else x4
        
        x4 = self.Conv4(x4)      # [B, 256, D/8, H/8, W/8]
        x5 = self.MaxPool(x4)
        x5 = self.dcla4([x1, x2, x3, x4], x5) if hasattr(self, 'dcla4') else x5
        
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

class ResUNetBaseline_S_SLKv2_MSF_v3(DCLA_UNet_v3):
    __remark__ = """
    [Version]: V2
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 712.220K
    • FLOPs: 47.264G
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True,
                 norm_type = 'batch',
                 act_type = 'relu',
                 ):
        super(ResUNetBaseline_S_SLKv2_MSF_v3, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear,
                norm_type =norm_type,
                act_type = act_type
        )
        del_list = ['dcla1', 'dcla2', 'dcla3', 'dcla4']
        for name in del_list:
            if hasattr(self, name):
                delattr(self, name)
        
    def forward(self, x):
        return super().forward(x) 

class ResUNetBaseline_S_SLKv2_v3(DCLA_UNet_v3):
    __remark__ = """
    [Version]: V2.3
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 1.211M
    • FLOPs: 102.428G
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True,
                 dropout_rate=0,
                 norm_type = 'batch',
                 act_type = 'relu',
                 ):
        super(ResUNetBaseline_S_SLKv2_v3, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate,
                norm_type =norm_type,
                act_type = act_type
        )
        del_list = ['dcla1', 'dcla2', 'dcla3', 'dcla4']
        for name in del_list:
            if hasattr(self, name):
                delattr(self, name)
                
        self.UpConv4 = ResConv3D_S_BN(f_list[3]*2, f_list[3]//2, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.UpConv3 = ResConv3D_S_BN(f_list[2]*2, f_list[2]//2, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.UpConv2 = ResConv3D_S_BN(f_list[1]*2, f_list[1]//2, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.UpConv1 = ResConv3D_S_BN(f_list[0]*2, f_list[0], dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
    def forward(self, x):
        return super().forward(x)

class ResUNetBaseline_S_DCLA_SLKv2_v3(DCLA_UNet_v3):
    __remark__ = """
    [Version]: V2.3
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 1.211M
    • FLOPs: 102.428G
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 dropout_rate=0,
                 norm_type = 'batch',
                 act_type = 'relu',
                 ):
        super(ResUNetBaseline_S_DCLA_SLKv2_v3, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate,
                norm_type =norm_type,
                act_type = act_type,
        )
        self.UpConv4 = ResConv3D_S_BN(f_list[3]*2, f_list[3]//2, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.UpConv3 = ResConv3D_S_BN(f_list[2]*2, f_list[2]//2, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.UpConv2 = ResConv3D_S_BN(f_list[1]*2, f_list[1]//2, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.UpConv1 = ResConv3D_S_BN(f_list[0]*2, f_list[0], dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
    def forward(self, x):
        return super().forward(x)
    
class ResUNetBaseline_S_DCLA_v3(DCLA_UNet_v3):
    __remark__ = """
    [Version]: V2
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
                    dropout_rate=0,
                    norm_type = 'batch',
                    act_type = 'relu',
                    ):
        super(ResUNetBaseline_S_DCLA_v3, self).__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                f_list=f_list,
                trilinear=trilinear,
                dropout_rate=dropout_rate,
                norm_type = norm_type,
                act_type = act_type
        )
        self.Conv1 = ResConv3D_S_BN(in_channels, f_list[0], dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.Conv2 = ResConv3D_S_BN(f_list[0], f_list[1], dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.Conv3 = ResConv3D_S_BN(f_list[1], f_list[2], dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.Conv4 = ResConv3D_S_BN(f_list[2], f_list[3], dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        
        self.UpConv4 = ResConv3D_S_BN(f_list[3]*2, f_list[3]//2, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.UpConv3 = ResConv3D_S_BN(f_list[2]*2, f_list[2]//2, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.UpConv2 = ResConv3D_S_BN(f_list[1]*2, f_list[1]//2, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.UpConv1 = ResConv3D_S_BN(f_list[0]*2, f_list[0], dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        
    def forward(self, x):
        return super().forward(x)    

class ResUNetBaseline_S_DCLA_MSF_v3(DCLA_UNet_v3):
    __remark__ = """
    [Version]: V2
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 3.027M
    • FLOPs: 153.810G
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 kernel_size=3, 
                 dropout_rate=0,
                 norm_type = 'batch',
                 act_type = 'relu',
                 ):
        super(ResUNetBaseline_S_DCLA_MSF_v3, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                kernel_size=kernel_size, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate,
                norm_type = norm_type,
                act_type = act_type,
        )
        self.Conv1 = ResConv3D_S_BN(in_channels, f_list[0], kernel_size=kernel_size, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.Conv2 = ResConv3D_S_BN(f_list[0], f_list[1], kernel_size=kernel_size, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.Conv3 = ResConv3D_S_BN(f_list[1], f_list[2], kernel_size=kernel_size, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.Conv4 = ResConv3D_S_BN(f_list[2], f_list[3], kernel_size=kernel_size, dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
    def forward(self, x):
        return super().forward(x)       

class ResUNetBaseline_S_MSF_v3(DCLA_UNet_v3):
    __remark__ = """
    [Version]: V2
    [Author]: Junyin Xiong
    [basline]: ResUNetBaseline_S
    [Features]
    • 总参数量: 3.027M
    • FLOPs: 153.810G
    """
    def __init__(self,
                 in_channels=4, 
                 out_channels=4,
                 f_list=[32, 64, 128, 256], 
                 trilinear=True, 
                 kernel_size=3, 
                 dropout_rate=0,
                 norm_type = 'batch',
                 act_type = 'relu',
                 ):
        super(ResUNetBaseline_S_MSF_v3, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels,
                f_list=f_list, 
                kernel_size=3, 
                trilinear=trilinear, 
                dropout_rate=dropout_rate,
                norm_type = norm_type,
                act_type = act_type,
        )
        del_list = ['dcla1', 'dcla2', 'dcla3', 'dcla4']
        for name in del_list:
            if hasattr(self, name):
                delattr(self, name)
        self.Conv1 = ResConv3D_S_BN(in_channels, f_list[0], dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.Conv2 = ResConv3D_S_BN(f_list[0], f_list[1], dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.Conv3 = ResConv3D_S_BN(f_list[1], f_list[2], dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
        self.Conv4 = ResConv3D_S_BN(f_list[2], f_list[3], dropout_rate=dropout_rate, norm_type=norm_type, act_type=act_type)
    def forward(self, x):
        return super().forward(x)   

if __name__ == "__main__":
    test_unet(model_class=DCLA_UNet_v3, batch_size=1)   
    model = DCLA_UNet_v3(in_channels=4, out_channels=4)
    print(model.__remark__)
    
    
