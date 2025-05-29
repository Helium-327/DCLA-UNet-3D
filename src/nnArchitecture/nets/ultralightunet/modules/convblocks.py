# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/26 18:13:02
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 复现UltralightUNet的编解码结构
*      VERSION: v1.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''


import torch
import torch.nn as nn
from .commons import *

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, norm_type="batch", act_type='relu6', dw_parallel=True):
        super(MultiKernelDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                norm_layer(name=norm_type, num_channels=self.in_channels),
                act_layer(act_type)
            )
            for kernel_size in kernel_sizes
        ])
    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        # For example, you could concatenate or add them; here, we just return the list
        return outputs

class MultiKernelInvertedResidualBlock(nn.Module):
    """
    inverted residual block used in MobileNetV2
    """
    def __init__(self, in_c, out_c, kernel_sizes=[1,3,5], norm_type="batch", act_type='relu6', dw_parallel=True, add=True):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        # check stride value
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)

        # expansion factor or t as mentioned in the paper
        self.ex_c = out_c
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv3d(self.in_c, self.ex_c, 1, 1, 0, bias=False), 
            nn.BatchNorm3d(self.ex_c),
            act_layer(act_type)
        )        
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(self.ex_c, self.kernel_sizes, 1, norm_type, act_type, dw_parallel=dw_parallel)

        if self.add == True:
            self.combined_channels = self.ex_c*1
        else:
            self.combined_channels = self.ex_c*self.n_scales
            
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv3d(self.combined_channels, self.out_c, 1, 1, 0, bias=False), # 
            nn.BatchNorm3d(self.out_c),
        )
        self.shortcut = nn.Conv3d(self.in_c, self.out_c, 1, 1, 0, bias=False) if self.in_c != self.out_c else nn.Identity()

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        if self.add == True:
            dout = 0
            for dwout in dwconv_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(dwconv_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_c))
        out = self.pconv2(dout)

        x = self.shortcut(x)
        return x+out



class DepthwiseAxialConv3d(nn.Module):
    """深度可分离轴向3D卷积（分XYZ三个轴向依次卷积）
    
    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 轴向卷积核尺寸，默认为3
        stride (int): 卷积步长，默认为1
        dilation (int): 膨胀系数，默认为None
        groups (int): 分组数，默认为None(深度可分离卷积)
        
    注意：
        - 当使用dilation时自动计算等效padding
        - 最后使用1x1卷积调整通道数
    """
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

def channel_shuffle(x, groups):
    batchs, channels, depth, height, width = x.data.size()
    channels_per_group = channels // groups
    
    # reshape
    x = x.view(batchs, groups, channels_per_group, depth, height, width)
    
    x = torch.transpose(x, 1, 2).contiguous()
    
    # flatten
    x = x.view(batchs, -1, depth, height, width)
    
    return x    

class ResDepthwiseAxialConv3d(nn.Module):
    """深度可分离轴向3D卷积（分XYZ三个轴向依次卷积）
    
    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 轴向卷积核尺寸，默认为3
        stride (int): 卷积步长，默认为1
        dilation (int): 膨胀系数，默认为None
        groups (int): 分组数，默认为None(深度可分离卷积)
        
    注意：
        - 当使用dilation时自动计算等效padding
        - 最后使用1x1卷积调整通道数
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 dilation=None, 
                 groups= None,
                 norm_type="batch",
                 act_type= "gelu",
                 use_residual= True
                 ):
        super().__init__()  # 首先调用父类的 __init__ 方法
        
        self.use_residual = use_residual
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
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = norm_layer(norm_type, out_channels)
        self.act =  act_layer(name=act_type)
        self.shortcut = nn.AdaptiveAvgPool3d(1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        
        out = self.AxialConv(x)
        out = self.pointwise(out)
        out = self.norm(out)
        out = self.act(out)
        
        if self.use_residual:
            x += self.shortcut(x)
        return x

class DepthAxialMultiScaleResidualBlock(nn.Module):  
    """深度可分离轴向多核残差块
    
    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size_list (list): 卷积核尺寸列表
        stride (int): 卷积步长，默认为1
        dilation (int): 膨胀系数，默认为None
        groups (int): 分组数，默认为None(深度可分离卷积)
        
    注意：
        - 当使用dilation时自动计算等效padding
    """
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 kernel_size=3, 
                 stride=1, 
                 dilation=[1,2], 
                 groups=None,
                 norm_type="batch",
                 act_type="gelu",
                 add=True,
                 
                 ):
        super().__init__()
        self.dilation = dilation
        self.add = add
        
        # 前置通道变换
        self._pwconv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, 1, 0, bias=False),
            norm_layer(name=norm_type, num_channels=out_ch),
            act_layer(name=act_type)
        )
        
        # 多尺度卷积
        self.add_module(
            name=f"multi_scale_convs",
            module=nn.ModuleList([
                ResDepthwiseAxialConv3d(
                in_channels=out_ch, 
                out_channels=out_ch, 
                kernel_size=kernel_size, 
                stride=stride, 
                dilation=dilation[i], 
                groups=groups,
                norm_type=norm_type,
                act_type=act_type
                ) for i in range(len(dilation))
                ]
            )
        )
        
        self.combined_ch = out_ch *1 if self.add else out_ch * len(dilation)
        self.channel_groups = len(dilation) if out_ch % len(dilation)==0 else 1
        # 后置通道变换
        self.pwconv_ = nn.Sequential(
            nn.Conv3d(self.combined_ch, out_ch, 1, 1, 0, bias=False),
            norm_layer(norm_type, out_ch)
        )
        self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = act_layer(name=act_type)
    def forward(self, x):
        residual = self.shortcut(x) 
        x = self._pwconv(x)
        ms_out = [conv(x) for conv in self.multi_scale_convs]
        if self.add:
            ms_out = torch.sum(torch.stack(ms_out), dim=0)
            x = channel_shuffle(ms_out, self.channel_groups)
            x = self.pwconv_(ms_out)
        else:
            ms_out = torch.cat(ms_out, dim=1)
            x = self.pwconv_(ms_out)
        x += residual
        return x

class DepthAxialMultiScaleResidualBlockv2(nn.Module):  
    """深度可分离轴向多核残差块
    
    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size_list (list): 卷积核尺寸列表
        stride (int): 卷积步长，默认为1
        dilation (int): 膨胀系数，默认为None
        groups (int): 分组数，默认为None(深度可分离卷积)
        
    注意：
        - 当使用dilation时自动计算等效padding
    """
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 kernel_size=[3,5,7], 
                 stride=1, 
                 dilation=[1,2], 
                 groups=None,
                 norm_type="batch",
                 act_type="gelu",
                 add=True,
                 
                 ):
        super().__init__()
        self.dilation = dilation
        self.add = add
        
        # 前置通道变换
        self._pwconv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, 1, 0, bias=False),
            norm_layer(name=norm_type, num_channels=out_ch),
            act_layer(name=act_type)
        )
        
        # 多尺度卷积
        self.add_module(
            name=f"multi_scale_convs",
            module=nn.ModuleList([
                ResDepthwiseAxialConv3d(
                in_channels=out_ch, 
                out_channels=out_ch, 
                kernel_size=k, 
                stride=stride, 
                groups=groups,
                norm_type=norm_type,
                act_type=act_type
                ) for k in kernel_size
                ]
            )
        )
        
        self.combined_ch = out_ch *1 if self.add else out_ch * len(dilation)
        self.channel_groups = len(dilation) if out_ch % len(dilation)==0 else 1
        # 后置通道变换
        self.pwconv_ = nn.Sequential(
            nn.Conv3d(self.combined_ch, out_ch, 1, 1, 0, bias=False),
            norm_layer(norm_type, out_ch)
        )
        self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = act_layer(name=act_type)
    def forward(self, x):
        residual = self.shortcut(x) 
        x = self._pwconv(x)
        ms_out = [conv(x) for conv in self.multi_scale_convs]
        if self.add:
            ms_out = torch.sum(torch.stack(ms_out), dim=0)
            x = channel_shuffle(ms_out, self.channel_groups)
            x = self.pwconv_(ms_out)
        else:
            ms_out = torch.cat(ms_out, dim=1)
            x = self.pwconv_(ms_out)
        x += residual
        return self.act(x)
            
        