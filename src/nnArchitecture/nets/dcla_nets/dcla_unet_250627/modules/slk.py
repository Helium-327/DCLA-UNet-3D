

import torch
import torch.nn as nn 
from .conv_blocks import DepthwiseAxialConv3d
from .commons import *

class ResDWConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 norm_type="batch",
                 act_type="gelu"
                 ):
        super().__init__()
        self.depthwise  = nn.Sequential(
                nn.Conv3d(
                    out_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    groups=out_channels
                ),
                get_norm(norm_type, out_channels),
                get_act(act_type),
                nn.Conv3d(
                    out_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    groups=out_channels
                ),
                get_norm(norm_type, out_channels),
                get_act(act_type),
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
        self.depthwise  = nn.Sequential(
            DepthwiseAxialConv3d(
                out_channels,
                out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                kernel_size=kernel_size
            ),
            get_norm(norm_type, out_channels),
            get_act(act_type),
            DepthwiseAxialConv3d(
                out_channels,
                out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                kernel_size=kernel_size
            ),
            get_norm(norm_type, out_channels),
            get_act(act_type),
            
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
                                        act_type=act_type
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
                                        act_type=act_type
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
    
