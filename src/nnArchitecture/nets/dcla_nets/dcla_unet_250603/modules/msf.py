from .attn import Swish, SwishECA
from .commons import get_norm, get_act
from .convblocks import *


class MutilScaleFusionBlock(nn.Module): #(MLP)
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                #  mid_channels=None,
                 kernel_size=3, 
                 dilations=[1,2,3], 
                 fusion_kernel=7,  # 尺度和编码器保持一致
                 ratio=2,               # 4比2好
                 norm_type="batch",
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations
        
        # 创建多个具有不同核的分离卷积分支
        self.sep_branchs = nn.ModuleList([
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
            
        self.fusion = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                kernel_size=1
            ),
            get_norm(norm_type, out_channels)
        ) 
        # 残差连接，如果输入输出通道数不同，使用1x1卷积调整；否则使用恒等映射
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = get_act(act_type)
        
    def forward(self, x):
        out = x
        out = torch.sum(torch.stack([sep(out) for sep in self.sep_branchs]), dim=0)  # 使用加法而不是拼接
        
        out = self.fusion(out)

        out += self.residual(x)
        return self.act(out)   
    
    
"""改进方案1"""
class DynamicScaleFusion(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 dilations=[1,2,3],
                 reduction_ratio=8,
                 norm_type="batch",
                 act_type="gelu"      
                 ):
        super().__init__()
        
        # 动态权重生成器
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, len(dilations)),
            nn.Softmax(dim=1)
        )
        
        # 共享基础卷积核
        self.ms_branchs = nn.ModuleList([
            nn.Sequential(
                DepthwiseAxialConv3d(
                    in_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=3, 
                    dilation=d
                ),
                get_norm(norm_type, out_channels),
                get_act(act_type),
            ) for d in dilations
        ])
        self.dilations = dilations

        self.conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels,  kernel_size=1),
            get_norm(norm_type, out_channels),
        )
        
        self.act = get_act(act_type)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        # 生成动态权重 [B, num_dilations]
        weights = self.weight_net(x)  # (B, 3)
        
        # 动态卷积计算
        outputs = []
        for i, conv in enumerate(self.ms_branchs):
            # 动态调整扩张率
            conv_out = conv(x)
            # 加权输出
            outputs.append(conv_out * weights[:, i].view(-1,1,1,1,1))
            
        dynamic_out = torch.sum(torch.stack(outputs), dim=0)
        
        out = self.conv(dynamic_out) + self.shortcut(x)
        
        return self.act(out)

"""改进方案2"""
class DynamicMutilScaleFusionBlock(MutilScaleFusionBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 动态权重调整层
        self.dynamic_weights = nn.Sequential(
            nn.Conv3d(self.out_channels*len(self.dilations), 
                     len(self.dilations), 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        residual = self.residual(x)
        
        # 并行计算各分支
        branch_outputs = [sep(x) for sep in self.sep_branchs]
        
        # 动态融合权重
        weight_map = self.dynamic_weights(
            torch.cat(branch_outputs, dim=1))  # (B,3,D,H,W)
        
        # 空间自适应加权融合
        weighted_output = torch.stack([
            out * weight_map[:,i:i+1] 
            for i, out in enumerate(branch_outputs)
        ], dim=0).sum(dim=0)
        
        # 后续处理
        out = self.fusion(weighted_output)
        
        return self.act(out + residual)
    