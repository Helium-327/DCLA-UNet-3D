from .conv_blocks import *


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
    
class MutilScaleFusionBlock(nn.Module): #(MLP)
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size=3, 
                 dilations=[1,2,3], 
                 norm_type="instance",
                 act_type="gelu"
                 ):
        """Multi kernel separable convolution fusion block
        
        Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int, optional): 卷积核大小. 默认为 3
        stride (int, optional): 卷积步长. 默认为 1
        dilations (list, optional): 空洞卷积的扩张率列表. 默认为 [1,2,3]
        groups (int, optional): 分组卷积的组数. 默认为 None
        """
        super().__init__()
    
        norm_layer = get_norm(norm_type, out_channels, instance_norm_affine=True) if norm_type=="instance" else get_norm(norm_type, out_channels)
        self._conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            norm_layer,
            get_act(act_type)
        )
        # 创建多个具有不同核的分离卷积分支
        self.res_efficient_ms_block = EfficientResMultiScaleBlock(
                    out_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    dilations=dilations, 
                    norm_type=norm_type,
                    act_type=act_type
                 )
        
        self.conv_ = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1),
            norm_layer,
            get_act(act_type)
        )
        
    def forward(self, x):
        x = self._conv(x)
        x = self.res_efficient_ms_block(x)
        x = self.conv_(x)
        return x   