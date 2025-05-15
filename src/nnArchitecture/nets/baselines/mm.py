
# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/15 16:34:03
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION:  各种模块
*      VERSION: v1.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch.nn as nn
import torch

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
    
class LightweightChannelAttention3Dv2(nn.Module):
    def __init__(self, kernel_size=3):
        super(LightweightChannelAttention3Dv2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.eca = nn.Sequential(
            nn.Conv1d(1, 1 , kernel_size=kernel_size, padding=kernel_size//2),
            Swish(),
            nn.Conv1d(1, 1 , kernel_size=kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, _, _, _, _ = x.size()
        avg_out = (self.avg_pool(x) + self.max_pool(x)).view(batch_size, -1)
        attn = self.eca(avg_out.unsqueeze(1)).view(batch_size, -1, 1, 1, 1)
        return attn
    
class LightweightSpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=3):
        super(LightweightSpatialAttention3D, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm3d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # x = avg_out + max_out
        x = self.bn(self.conv1(x))
        return self.sigmoid(x)

class LightweightChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(LightweightChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.act = Swish()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc2(self.act(self.fc1(self.avg_pool(x) + self.max_pool(x))))
        return self.sigmoid(out)
    
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
    
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
    
class ResConv3D(nn.Module):
    """带残差连接的各向异性卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.shortcut:
            residual = self.shortcut(residual)
        out += residual
        return self.relu(out)

class AttentionBlock3D(nn.Module):
    """3D Attention Gate"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi