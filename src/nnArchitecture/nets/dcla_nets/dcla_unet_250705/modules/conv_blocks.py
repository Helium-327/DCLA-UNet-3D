import torch
import torch.nn as nn 

from .commons import get_act, get_norm

class ResSLK(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 norm_type="batch",
                 act_type="gelu"
                 ):
        super().__init__()
        self.depthwise = nn.Sequential(
                DepthwiseAxialConv3d(
                    out_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size
                ),
                get_norm(norm_type, out_channels),
                get_act(act_type),
                # DepthwiseAxialConv3d(
                #     out_channels,
                #     out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                #     kernel_size=kernel_size
                # ),
                # get_norm(norm_type, out_channels),
                # get_act(act_type)
            )
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        return self.depthwise(x) + self.residual(x)
    
class EfficientResMultiScaleBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 dilations=[1,2,3], 
                 norm_type="batch",
                 act_type="gelu"
                 ):
        super().__init__()
        self.ms_branchs = nn.ModuleList([
            nn.Sequential(
                DepthwiseAxialConv3d(
                    in_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size, 
                    dilation=d
                ),
                get_norm(norm_type, out_channels),
                get_act(act_type),
            ) for d in dilations
        ])
        
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        
    def forward(self, x):
        residual = self.residual(x)
        out = torch.sum(torch.stack([sep(x) for sep in self.ms_branchs]), dim=0)  # 使用加法而不是拼接
        return out + residual
    
    
class MSLKBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size=7, 
                 dilations=[1,2,3], 
                 norm_type="batch",
                 act_type="gelu",
                 ms_first=False,
                 slk=True,
                 msf=True
                 ):
        super(MSLKBlock, self).__init__()
        self.slk = slk
        self.msf = msf
        self.ms_first = ms_first
        self.add_module(
            "_conv", 
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1),
                get_norm(norm_type, out_channels),
                get_act(act_type)
            )
        )
        
        if self.msf:
            self.add_module(
                "msf_block", 
                EfficientResMultiScaleBlock(
                    out_channels, 
                    out_channels, 
                    kernel_size=3, 
                    dilations=dilations, 
                    norm_type=norm_type,
                    act_type=act_type
                )
        )
        
        if self.slk:
            self.add_module(
                "slk_block", 
                ResSLK(
                    out_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    norm_type=norm_type,
                    act_type=act_type
                )
            )
        
        self.add_module(
            "conv_",
            nn.Sequential(
                nn.Conv3d(out_channels, out_channels, 1),
                get_norm(norm_type, out_channels),
                get_act(act_type)
            )
        )
        
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        out = self._conv(x)
        if self.ms_first:
            out = self.msf_block(out)
            out = self.slk_block(out)
        else:
            out = self.slk_block(out)
            out = self.msf_block(out)
            
        out = self.conv_(out)
        
        return self.shortcut(x) + out


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
    
    
class ResDAConv3d(nn.Module):
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
                 act_type="gelu",
                 ):
        super().__init__()  # 首先调用父类的 __init__ 方法
        if dilation is not None:
            k_eq = (kernel_size - 1)*dilation + 1
            p_eq = (k_eq - 1)//2
            assert k_eq % 2 == 1, "kernel_size must be odd"
        self.avg = nn.AdaptiveAvgPool3d(1)
        
        self.AxialConv = nn.ModuleList([
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
        ]
        )
        # 添加一个点卷积来改变通道数
        self.pointwise = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            get_norm(norm_type, out_channels),
            get_act(act_type)
            )
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        x_avg = self.avg(x)
        out = [conv(x) for conv in self.AxialConv]
        x = torch.stack(out, dim=0).sum(dim=0)
        x = x_avg + x  # 残差连接
        
        x = self.pointwise(x)
        # x = self.dropout(x)
        return x
    
    
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
    
class ResNeXtConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_rate=2,
        kernel_size=3,
        norm_type="batch",
        act_type="gelu",
        dropout=0.2,
    ):
        super().__init__()
        self.stride = stride
        
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.conv_list = nn.ModuleList()

        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels * expand_rate, 1, 1, 0),
                get_norm(norm_type, in_channels * expand_rate, instance_norm_affine=True),
                get_act(act_type)
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels * expand_rate,
                    in_channels * expand_rate,
                    kernel_size,
                    stride,
                    kernel_size//2,
                    groups=in_channels,
                ),
                get_norm(norm_type, in_channels * expand_rate, instance_norm_affine=True),
                get_act(act_type),
                nn.Dropout3d(dropout),
                nn.Conv3d(
                    in_channels * expand_rate,
                    in_channels * expand_rate,
                    kernel_size,
                    stride,
                    kernel_size//2,
                    groups=in_channels,
                ),
                get_norm(norm_type, in_channels * expand_rate, instance_norm_affine=True),
                get_act(act_type)
            )
        )  
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels * expand_rate, out_channels, 1, 1, 0),
                get_norm(norm_type, out_channels, instance_norm_affine=True),
                get_act(act_type)
            )
        )
        
        self.residual = in_channels == out_channels

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.trunc_normal_(m.weight, std=0.06)
        #         nn.init.constant_(m.bias, 0)
    def forward(self, x):
        res = x
        x = self.conv_list[0](x) 
        avg = self.avg(x)
        x = self.conv_list[1](x) 
        x += avg
        x = self.conv_list[2](x) 
        x = x + res if self.residual and self.stride == 1 else x
        return x

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