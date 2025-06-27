

import torch
import torch.nn as nn 
from .conv_blocks import DepthwiseAxialConv3d
from .commons import *

class SlimLargeKernelBlock(nn.Module): 
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 ratio=4,
                 depth=2 
                ):
        super().__init__()
        self.depthwise = nn.Sequential(
                DepthwiseAxialConv3d(
                    in_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size
                ),
                nn.BatchNorm3d(out_channels),
                nn.GELU(),
                DepthwiseAxialConv3d(
                    out_channels,
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
    
#! 改进点1 更换编码器

import math
class SwishECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(SwishECA, self).__init__()
        
        # 创建一个自适应平均池化层和自适应最大池化层
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 动态计算1D卷积核大小 (保持原论文公式)
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        # 使用1D卷积进行通道注意力
        self.ca = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            Swish(),
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.shape
        attn = self.ca((self.avg_pool(x) + self.max_pool(x)).view(b, 1, -1)).view(b, -1, 1, 1, 1)
        return attn * x

class ResBlockOfDepthwiseAxialConv3D(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 use_act = True,
                 norm_type="batch",
                 act_type = 'gelu',
                ):
        super().__init__()
        self.use_act = use_act
        self.conv = nn.Sequential(
                DepthwiseAxialConv3d(
                    in_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size
                ),
                get_norm(norm_type, out_channels),
        )
        self.act = get_act(act_type)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        out = self.conv(x)
        out += self.residual(x)
        return self.act(out)
    
class SlimLargeKernelBlockv1(nn.Module): 
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                ):
        super().__init__()
        
        self.depthwise = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=1),
            DepthwiseAxialConv3d(
                    out_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size
                ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            # SwishECA(out_channels),
            DepthwiseAxialConv3d(
                    out_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size
                ),
            nn.BatchNorm3d(out_channels),
        )
        
        self.act = nn.GELU() 
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        out = self.depthwise(x) + self.residual(x)
        return self.act(out)
    
class ResMLP(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 ratio=4,
                 reduce=True
                ):
        super().__init__()
        
        mid_channels = in_channels // ratio if reduce else in_channels * ratio
        
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm3d(mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels,out_channels,kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()
    def forward(self, x):
        out = self.mlp(x) + self.residual(x)
        return self.act(out)
        
class ResMLPv2(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 ratio=4,
                 reduce=True,
                 norm_type="batch",
                 act_type="gelu"
                ):
        super().__init__()
        
        mid_channels = in_channels // ratio if reduce else in_channels * ratio
        
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=1),
            get_norm(norm_type, mid_channels),
            get_act(act_type),
            nn.Conv3d(mid_channels,out_channels,kernel_size=1),
            get_norm(norm_type, out_channels),
        )
        
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = get_act(act_type)
    def forward(self, x):
        out = self.mlp(x) + self.residual(x)
        return self.act(out)


class ResMLPv2(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 ratio=4,
                 reduce=True,
                 norm_type="batch",
                 act_type="gelu"
                ):
        super().__init__()
        
        mid_channels = in_channels // ratio if reduce else in_channels * ratio
        
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False),
            get_norm(norm_type, mid_channels),
            get_act(act_type),
            nn.Conv3d(mid_channels,out_channels,kernel_size=1, bias=False),
            get_norm(norm_type, out_channels),
        )
        
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = get_act(act_type)
    def forward(self, x):
        out = self.mlp(x) + self.residual(x)
        return self.act(out)

class SlimLargeKernelBlockv2(nn.Module): 
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                ):
        super().__init__()
        
        self.depthwise = nn.Sequential(
            ResBlockOfDepthwiseAxialConv3D(in_channels, out_channels, kernel_size),
            SwishECA(out_channels),
            ResBlockOfDepthwiseAxialConv3D(out_channels, out_channels, kernel_size),
            ResMLP(out_channels, out_channels),
        )
        
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()
    def forward(self, x):
        out = self.depthwise(x) + self.residual(x)
        return self.act(out)

class SlimLargeKernelBlockv3(nn.Module): 
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 norm_type="batch",
                 act_type="gelu"
                ):
        super().__init__()
        
        self.depthwise = nn.Sequential(
            ResBlockOfDepthwiseAxialConv3D(in_channels, out_channels, kernel_size, norm_type=norm_type, act_type=act_type),
            ResBlockOfDepthwiseAxialConv3D(out_channels, out_channels, kernel_size, norm_type=norm_type, act_type=act_type),
            SwishECA(out_channels, gamma=1, b=2),
        )
        
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = get_act(act_type)
    def forward(self, x):
        out = self.depthwise(x) + self.residual(x)
        return self.act(out)  
    
class ResDWConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 norm_type="batch",
                 act_type="gelu",
                 depth=2
                 ):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    out_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    groups=out_channels
                ),
                get_norm(norm_type, out_channels),
                get_act(act_type)
            ) for _ in range(depth)
        ])
        
        self.depthwise  = nn.Sequential(
            *self.conv_list
        )
        
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()        
    
    def forward(self, x):
        return self.depthwise(x) + self.residual(x)
        
        
class ResDWAConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 norm_type="batch",
                 act_type="gelu",
                 depth=2
                 ):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Sequential(
                DepthwiseAxialConv3d(
                    out_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size
                ),
                get_norm(norm_type, out_channels),
                get_act(act_type)
            ) for _ in range(depth)
        ])
        
        self.depthwise  = nn.Sequential(
            *self.conv_list
        )
        
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        return self.depthwise(x) + self.residual(x)
        
class SlimLargeKernelBlockv4(nn.Module): 
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 norm_type="batch",
                 act_type="gelu",
                 depth=2,
                 dropout_rate=0.1
                ):
        super().__init__()
        
        self._conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            get_norm(norm_type, out_channels),
            get_act(act_type)
        )
        
        self.resblock = ResDWAConvBlock(out_channels, 
                                        out_channels, 
                                        kernel_size=kernel_size, 
                                        norm_type=norm_type, 
                                        act_type=act_type,
                                        depth=depth
                                        )
        
        self.conv_ = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1),
            get_norm(norm_type, out_channels),
            get_act(act_type)
        )
        
    def forward(self, x):
        out = self._conv(x)
        # map = self.avg(out)
        out = self.resblock(out)
        # out = map + out
        out = self.conv_(out)
        return out
    
    
class SlimLargeKernelBlockv5(nn.Module): 
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 norm_type="batch",
                 act_type="gelu",
                 depth=2,
                 dropout_rate=0.1
                ):
        super().__init__()
        
        self._conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            get_norm(norm_type, out_channels),
            get_act(act_type)
        )
        
        self.resblock = ResDWConvBlock(out_channels, 
                                        out_channels, 
                                        kernel_size=kernel_size, 
                                        norm_type=norm_type, 
                                        act_type=act_type,
                                        depth=depth
                                        )
        
        self.conv_ = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1),
            get_norm(norm_type, out_channels),
            get_act(act_type)
        )
        
    def forward(self, x):
        out = self._conv(x)
        out = self.resblock(out)
        out = self.conv_(out)
        return out  
    
