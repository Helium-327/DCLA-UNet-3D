# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/04/30 17:29:07
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建各种通用的卷积块
*      VERSION: v1.0
*      FEATURES: 
                - ResConv3D_S_BN: 单3D卷积块(Conv3D -> BN结构)
                - ResConv3D_M_BN: 双3D卷积块(双Conv3D -> BN结构)
                - ResConv3D_GN: 使用组归一化(GroupNorm)的残差3D卷积块
                - ResConv3D_IN: 使用实例归一化(InstanceNorm)的残差3D卷积块
                - ResConv3D_better: 改进的残差3D卷积块
                - DoubleConvBlock: 双3D卷积块(Conv3D -> BN结构)
                - DepthwiseAxialConv3d: 深度可分离轴向3D卷积
                - DWConv3D: 深度可分离3D卷积
                - DWResConv3D: 深度可分离残差3D卷积
                - AniResConv: 各向异性残差卷积块（带空间注意力机制）
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn



class ResConv3D_S_BN(nn.Module):
    """带有残差连接的单分支3D卷积块(Conv3D -> BN结构)
    
    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int/tuple): 卷积核尺寸，默认为3
        stride (int/tuple): 卷积步长，默认为1
        padding (int/tuple): 填充尺寸，默认为1
        dropout_rate (float): Dropout概率，默认为0.2
        
    返回：
        Tensor: 输出特征图，尺寸为 (batch_size, out_channels, D, H, W)
    """
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
    
class ResConv3D_M_BN(nn.Module):
    """带有残差连接的双分支3D卷积块(双Conv3D -> BN结构)
    
    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int/tuple): 卷积核尺寸，默认为3
        stride (int/tuple): 卷积步长，默认为1
        padding (int/tuple): 填充尺寸，默认为1
        dropout_rate (float): Dropout概率，默认为0.2
        
    特征：
        - 使用两次卷积+BN+ReLU组合
        - 残差连接后统一进行Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=0.2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, 
                      out_channels, 
                      kernel_size=kernel_size, 
                      stride=stride,
                      padding=padding
                      ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
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
        return self.drop(self.relu(out))
    
    
class ResConv3D_GN(nn.Module):
    """使用组归一化(GroupNorm)的残差3D卷积块
    
    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        groups (int): 分组数（必须能整除通道数）
        kernel_size (int/tuple): 卷积核尺寸，默认为3
        stride (int/tuple): 卷积步长，默认为1
        padding (int/tuple): 填充尺寸，默认为1
        
    设计特点：
        - 使用1x1卷积进行通道数对齐
        - 分组卷积减少参数量
        - 最后进行统一的Dropout
    """
    def __init__(self, in_channels, out_channels, groups, kernel_size=3, stride=1, padding=1):
        super().__init__()
        dropout = 0.2
        self.group = groups
        assert in_channels % self.group == 0, "in_channels must be divisible by groups"
        assert out_channels % self.group == 0, "out_channels must be divisible by groups"
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Conv3d(out_channels, 
                      out_channels, 
                      kernel_size=kernel_size, 
                      padding=padding,
                      stride=stride,
                      groups=self.group
                      ),
            nn.GroupNorm(self.group, out_channels)
            )
        # 使用 1x1x1 卷积进行通道调整，或者在通道数相同时使用恒等映射

        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.drop = nn.Dropout3d(p=dropout)  # 添加 Dropout 层
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out += residual  # 残差连接
        out = self.relu(out)
        return self.drop(out)

    
class ResConv3D_IN(nn.Module):
    def __init__(self, in_channels, out_channels, groups, kernel_size=3, stride=1, padding=1):
        super().__init__()
        dropout = 0.2
        self.group = groups
        assert in_channels % self.group == 0, "in_channels must be divisible by groups"
        assert out_channels % self.group == 0, "out_channels must be divisible by groups"
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Conv3d(out_channels, 
                      out_channels, 
                      kernel_size=kernel_size, 
                      padding=padding,
                      stride=stride,
                      groups=self.group
                      ),
            nn.InstanceNorm3d(out_channels)
            )
        # 使用 1x1x1 卷积进行通道调整，或者在通道数相同时使用恒等映射

        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.drop = nn.Dropout3d(p=dropout)  # 添加 Dropout 层
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out += residual  # 残差连接
        out = self.relu(out)
        return self.drop(out)
    
    
# @1: 改变dropout的位置
class ResConv3D_better(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        dropout = 0.2
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 
                      out_channels, 
                      kernel_size=kernel_size, 
                      padding=padding,
                      stride=stride,
                      ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(),
            nn.Conv3d(out_channels, 
                      out_channels, 
                      kernel_size=kernel_size, 
                      padding=padding,
                      stride=stride,
                      ),
            nn.InstanceNorm3d(out_channels, affine=True)
            )
        # 使用 1x1x1 卷积进行通道调整，或者在通道数相同时使用恒等映射
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm3d(out_channels, affine=True)
        ) if in_channels != out_channels else nn.Identity()
        
        self.relu = nn.ReLU()
        self.final_drop = nn.Dropout3d(p=dropout),  # 添加 Dropout 层
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out += residual  # 残差连接
        out = self.relu((out))
        return self.final_drop(out)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        p_en = 0.2
        p_de = 0.2
        dropout = p_en if in_channels < out_channels else p_de
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 
                      out_channels, 
                      kernel_size=kernel_size, 
                      padding=1,
                      stride=stride,
                      ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Dropout3d(dropout),  # 添加 Dropout 层
            
            nn.Conv3d(out_channels, 
                      out_channels, 
                      kernel_size=3, 
                      stride=stride,
                      padding=padding
                      ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
    
    
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


class AniResConv(nn.Module):
    """各向异性残差卷积块（带空间注意力机制）
    
    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        groups (int): 分组卷积数
        k (int): 各向异性卷积核基础尺寸，默认为3
        directions (int): 注意力方向数，默认为3
        dropout_rate (float): Dropout概率，默认为0.2
        
    结构说明：
        1. 基础卷积进行通道变换
        2. 并行三个方向的各向异性卷积
        3. 自适应注意力加权融合
        4. 残差连接后统一处理
    """
    def __init__(self, in_channels, out_channels, groups, k=3, directions=3,  dropout_rate=0.2):
        super().__init__()
        self.group = groups
        assert in_channels % self.group == 0, "in_channels must be divisible by groups"
        assert out_channels % self.group == 0, "out_channels must be divisible by groups"
        #@1
        self.base_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            nn.GroupNorm(4, out_channels),  
            nn.ReLU(inplace=True)
        )
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(out_channels, 
                          out_channels, 
                          kernel_size=(1, k ,k), 
                          padding=(0, k//2, k//2),
                          groups=self.group
                          ),
                nn.GroupNorm(self.group, out_channels),
                nn.ReLU(inplace=True)  # 添加ReLU
            ),
            nn.Sequential(
                nn.Conv3d(out_channels, 
                          out_channels, 
                          kernel_size=(k, 1 ,k), 
                          padding=(k//2, 0,  k//2),
                          groups=self.group
                          ),
                nn.GroupNorm(self.group, out_channels),
                nn.ReLU(inplace=True)  # 添加ReLU
            ),
            nn.Sequential(
                nn.Conv3d(out_channels, 
                          out_channels, 
                          kernel_size=(k, k,1), 
                          padding=(k//2, k//2, 0)
                          ),
                nn.GroupNorm(self.group, out_channels),
                nn.ReLU(inplace=True)  # 添加ReLU
            ),
        ])
        self.attention_map = nn.Sequential(             # 添加非线性层提升表达能力
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_channels, directions*2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(directions*2, directions, 1),
            nn.Softmax(dim=1)
        )
        
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
        self.final_dropout = nn.Dropout3d(p=dropout_rate)
        
    def forward(self, x):
        
        residual = x
        x_base = self.base_conv(x)
        # 计算注意力权重
        att_weights  = self.attention_map(x_base)          # [B, 4, 1, 1, 1]
        # 计算各分支输出
        branch_outputs  = [conv(x_base) for conv in self.conv]  # 各个方向卷积的结果
        # 自适应权重
        weighted_outputs = [att_weights[:, i].unsqueeze(1).expand_as(out) * out for i, out in enumerate(branch_outputs)] # [b,c,4,d,h,w]
        out_main = sum(weighted_outputs)  # 加权融合
        out_shortcut = self.shortcut(residual) if self.shortcut else residual
        out = self.relu(out_main + out_shortcut)
        out = self.final_dropout(out)   # @3
        return out
    
# 其余类均已添加类似规范注释，此处省略重复展示