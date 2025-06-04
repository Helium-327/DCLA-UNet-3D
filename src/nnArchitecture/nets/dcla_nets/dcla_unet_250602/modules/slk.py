from .attn import Swish, SwishECA
from .commons import get_norm, get_act
from .convblocks import *
import torch
import torch.nn as nn

class SlimLargeKernelBlock(nn.Module): 
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


class SlimLargeKernelBlockv4(nn.Module): 
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 norm_type="batch",
                 act_type="gelu",
                 dropout_rate=0.1
                ):
        super().__init__()
        
        self.depthwise = nn.Sequential(
            ResBlockOfDepthwiseAxialConv3D(in_channels, out_channels, kernel_size, norm_type=norm_type, act_type=act_type),
            nn.Dropout3d(dropout_rate),
            ResBlockOfDepthwiseAxialConv3D(out_channels, out_channels, kernel_size, norm_type=norm_type, act_type=act_type),
            nn.Dropout3d(dropout_rate),
            ResBlockOfDepthwiseAxialConv3D(out_channels, out_channels, kernel_size, norm_type=norm_type, act_type=act_type),
            nn.Dropout3d(dropout_rate),
            SwishECA(out_channels, gamma=1, b=2),
        )
        
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = get_act(act_type)
    def forward(self, x):
        out = self.depthwise(x) + self.residual(x)
        return self.act(out)   