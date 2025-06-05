
import torch
import torch.nn as nn 


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

class SqueezeExcitation(nn.Module):
    """标准SE模块 (arXiv:1709.01507)"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, reduced_channels, kernel_size=1),
            Swish(),  # 使用现有Swish激活
            nn.Conv3d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.se(x) * x


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

class ResNeXtConv3(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_rate=2,
        kernel_size=3,
        norm_type="batch",
        act_type="gelu",
    ):
        super().__init__()
        self.stride = stride
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
                DepthwiseAxialConv3d(
                    in_channels * expand_rate,
                    in_channels * expand_rate,
                    kernel_size,
                    stride,
                    groups=in_channels,
                ),
                get_norm(norm_type, in_channels * expand_rate, instance_norm_affine=True),
                get_act(act_type)
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels * expand_rate, out_channels, 1, 1, 0),
                get_norm(norm_type, in_channels * expand_rate, instance_norm_affine=True),
                get_act(act_type)
            )
        )
        
        self.attn = SwishECA(out_channels)
        self.residual = in_channels == out_channels

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        res = x
        x = self.conv_list[0](x) 
        x = self.conv_list[1](x) 
        x = self.attn(x)
        x = self.conv_list[2](x)  
        x = x + res if self.residual and self.stride == 1 else x
        return x

# class SwishECA(nn.Module):
#     def __init__(self, channels, gamma=2, b=1):
#         super(SwishECA, self).__init__()
        
#         # 创建一个自适应平均池化层和自适应最大池化层
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.max_pool = nn.AdaptiveMaxPool3d(1)
        
#         # 动态计算1D卷积核大小 (保持原论文公式)
#         kernel_size = int(abs((math.log2(channels) + b) / gamma))
#         kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
#         # 使用1D卷积进行通道注意力
#         self.ca = nn.Sequential(
#             nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
#             Swish(),
#             nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _, _ = x.shape
#         attn = self.ca((self.avg_pool(x) + self.max_pool(x)).view(b, 1, -1)).view(b, -1, 1, 1, 1)
#         return attn * x

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

class MutilScaleFusionBlock(nn.Module): #(MLP)
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 dilations=[1,2,3], 
                 norm_type="batch",
                 act_type="gelu",
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

class MutilScaleFusionBlockv2(nn.Module): #(MLP)
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                #  mid_channels=None,
                 kernel_size=3, 
                 dilations=[1,2,3], 
                 fusion_kernel=7,  # 尺度和编码器保持一致
                 ratio=8,               # 4比2好
                 norm_type="batch",
                 act_type="gelu",
                 use_attn=True,
                 use_fusion=True
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
        
        self.use_attn = use_attn
        self.use_fusion = use_fusion
        # 创建多个具有不同核的分离卷积分支
        self.sep_branchs = nn.ModuleList([
                ResDAConv3d(
                    in_channels,
                    out_channels,  # 每个分支的输出通道数为总输出通道数的一半
                    kernel_size=kernel_size, 
                    dilation=d
                )
            for d in dilations
        ])
            
        if use_fusion: 
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
            
        if self.use_fusion:
            out = self.fusion(out)

        out += self.residual(x)
        return self.act(out)    

class AdaptiveSpatialCondenser(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 kernel_size=[7,5,3], 
                 in_size=128, 
                 min_size=8,
                 fusion_mode='concat'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.in_size = in_size
        self.min_size = min_size
        self.fusion_mode = fusion_mode
        self.branches = self._build_multi_branchs()  # 动态构建下采样序列
        
    def _build_multi_branchs(self):
        return nn.ModuleList([self._build_single_branch(k) for k in self.kernel])
        
    def _build_single_branch(self, kernel_size):
        layers = []
        current_size = self.in_size
        
        # 动态构建下采样序列
        while current_size > self.min_size:
            layers.append(
                nn.Sequential(
                nn.Conv3d(
                    self.in_channels, 
                    self.out_channels, 
                    kernel_size=kernel_size, 
                    stride=2, 
                    padding=kernel_size//2
                    ),
                nn.InstanceNorm3d(self.out_channels, affine=True),
                nn.GELU()
            ))
            current_size = current_size // 2
        return nn.Sequential(*layers)
            
    def forward(self, x):
        branch_ouputs = [branch(x) for branch in self.branches]
        
        if self.fusion_mode == 'concat':
            return torch.cat(branch_ouputs, dim=1)
        elif self.fusion_mode == 'add':
            return torch.sum(torch.stack(branch_ouputs), dim=0)
        else:
            raise ValueError("Invalid fusion mode. Choose from 'concat' or 'add'.")

class DynamicCrossLevelAttention(nn.Module): #MSFA
    def __init__(self, 
                 ch_list, 
                 feats_size, 
                 min_size=8, 
                 squeeze_kernel=1,
                 down_kernel=[3,5,7], 
                 fusion_kernel=1,
                 fusion_mode='add'
                 ):
        """
        Args: 
            ch_list: 输入特征的通道数
            feats_size: 输入特征的空间尺寸
            min_size: 最小空间尺寸
            kernel_size: 卷积核大小, 可以是一个整数或一个元组 (k1, k2, k3, k4)
        """
        super().__init__()
        self.ch_list = ch_list
        self.feats_size = feats_size
        self.min_size = min_size
        self.kernel_size = down_kernel if isinstance(down_kernel, list) else [down_kernel]
        self.fusion_mode = fusion_mode
        self.squeeze_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        
        # if isinstance(self.kernel_size, int):
        for ch in self.ch_list:
            self.squeeze_layers.append(
                nn.Sequential(
                    nn.Conv3d(ch, 1, kernel_size=squeeze_kernel, padding=squeeze_kernel//2),
                    nn.InstanceNorm3d(1, affine=True),
                    nn.GELU()
                    ))
        for feat_size in feats_size:
            self.down_layers.append(
                AdaptiveSpatialCondenser(
                    in_channels=1, 
                    out_channels=1, 
                    kernel_size=self.kernel_size, 
                    in_size=feat_size, 
                    min_size=8,
                    fusion_mode=self.fusion_mode  # 'concat' or 'add'
                )
            )
        self.conv = nn.Sequential(
            nn.Conv3d(len(self.kernel_size),
                      1, 
                      kernel_size=1, 
                      padding=0
                      ),
            nn.InstanceNorm3d(1, affine=True),
            nn.GELU(),
            nn.Conv3d(1, 1, kernel_size=1, padding=0)
        )
        self.fusion = nn.Conv3d(len(self.ch_list), 1, kernel_size=fusion_kernel, padding=fusion_kernel//2)

    def forward(self, encoder_feats, x):
        squeezed_feats = []
        
        # 压缩通道数
        for i , squeeze_layer in enumerate(self.squeeze_layers):
            squeezed_feats.append(squeeze_layer(encoder_feats[i]))

        downs = []
        
        # 压缩空间维度
        for i, feat in enumerate(squeezed_feats):
            need_down = (feat.size(2) != self.min_size)
            if need_down:
                down_feat = self.down_layers[i](feat)
            else:
                down_feat = feat
            if self.fusion_mode == 'concat':
                down_feat = self.conv(down_feat)
            elif self.fusion_mode == 'add':
                down_feat = down_feat
            else:
                raise ValueError("Invalid fusion mode. Choose from 'concat' or 'add'.")
            
            downs.append(down_feat.squeeze(1))
        # 特征融合
        fused = self.fusion(torch.stack(downs, dim=1))
        attn = torch.sigmoid(fused)
        
        out = attn * x
        return out
    
class DynamicCrossLevelAttentionv1(nn.Module): #MSFA
    def __init__(self, 
                 ch_list, 
                 feats_size, 
                 min_size=8, 
                 squeeze_kernel=3,
                 down_kernel=[3,5,7], 
                 fusion_kernel=1,
                 fusion_mode='add',
                 groups=4
                 ):
        """
        Args: 
            ch_list: 输入特征的通道数
            feats_size: 输入特征的空间尺寸
            min_size: 最小空间尺寸
            kernel_size: 卷积核大小, 可以是一个整数或一个元组 (k1, k2, k3, k4)
        """
        super().__init__()
        self.ch_list = ch_list
        self.feats_size = feats_size
        self.min_size = min_size
        self.kernel_size = down_kernel if isinstance(down_kernel, list) else [down_kernel]
        self.fusion_mode = fusion_mode
        self.squeeze_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        
        # if isinstance(self.kernel_size, int):
        for ch in self.ch_list:
            self.squeeze_layers.append(
                #! 改进点2：添加分组注意力
                nn.Sequential(
                    nn.Conv3d(ch, ch//groups, kernel_size=squeeze_kernel, padding=squeeze_kernel//2, groups=groups),
                    nn.GroupNorm(groups, ch//groups),
                    nn.GELU(),
                    nn.Conv3d(ch//groups, 1, kernel_size=1),
                    nn.InstanceNorm3d(1, affine=True),
                    nn.GELU()
                    ))
        for feat_size in feats_size:
            self.down_layers.append(
                AdaptiveSpatialCondenser(
                    in_channels=1, 
                    out_channels=1, 
                    kernel_size=self.kernel_size, 
                    in_size=feat_size, 
                    min_size=8,
                    fusion_mode=self.fusion_mode  # 'concat' or 'add'
                )
            )
        self.conv = nn.Sequential(
            nn.Conv3d(len(self.kernel_size),
                      1, 
                      kernel_size=1, 
                      padding=0
                      ),
            nn.InstanceNorm3d(1, affine=True),
            nn.GELU(),
            nn.Conv3d(1, 1, kernel_size=1, padding=0)
        )
        self.fusion = nn.Conv3d(len(self.ch_list), 1, kernel_size=fusion_kernel, padding=fusion_kernel//2)

    def forward(self, encoder_feats, x):
        squeezed_feats = []
        
        # 压缩通道数
        for i , squeeze_layer in enumerate(self.squeeze_layers):
            squeezed_feats.append(squeeze_layer(encoder_feats[i]))

        downs = []
        
        # 压缩空间维度
        for i, feat in enumerate(squeezed_feats):
            need_down = (feat.size(2) != self.min_size)
            if need_down:
                down_feat = self.down_layers[i](feat)
            else:
                down_feat = feat
            if self.fusion_mode == 'concat':
                down_feat = self.conv(down_feat)
            elif self.fusion_mode == 'add':
                down_feat = down_feat
            else:
                raise ValueError("Invalid fusion mode. Choose from 'concat' or 'add'.")
            
            downs.append(down_feat.squeeze(1))
        # 特征融合
        fused = self.fusion(torch.stack(downs, dim=1))
        attn = torch.sigmoid(fused)
        
        out = attn * x
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
        return self.sigmoid(x) * x


class SqueezeExcitation(nn.Module):
    """标准SE模块 (arXiv:1709.01507)"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(),  # 使用现有Swish激活
            nn.Conv3d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
    
class EfficientChannelAttention(nn.Module):
    """标准ECA模块 (arXiv:1910.03151)"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # 动态卷积核计算（保持与原论文一致）
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.conv = nn.Conv1d(1, 1, 
                            kernel_size=kernel_size,
                            padding=kernel_size//2, 
                            bias=False)
    def forward(self, x):
        y = self.avg_pool(x).flatten(1)  # [B, C, 1, 1, 1]
        # y = y.squeeze(-1).squeeze(-1).squeeze(-1)  # [B, C]
        y = y.unsqueeze(1)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 恢复形状
        return x * torch.sigmoid(y)    

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
        return self.sigmoid(out) * x
    
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
    
def get_norm(name: str, num_channels: int, **kwargs):
    """
    获取指定类型的归一化层
    
    Args:
        name (str): 归一化类型 ('batch', 'instance', 'group', 'layer')
        num_channels (int): 输入通道数
        **kwargs: 其他参数（如 group_norm_groups）
    
    Returns:
        nn.Module: 对应的归一化层
    """
    if name == 'batch':
        return nn.BatchNorm3d(num_channels)
    elif name == "instance":
        affine = kwargs.get("instance_norm_affine", True)
        return nn.InstanceNorm3d(num_channels, affine=affine)
    elif name == "group":
        groups = kwargs.get("group_norm_groups", 4)  # 默认 4 组
        assert num_channels % groups == 0, "通道数必须能被组数整除"
        return nn.GroupNorm(groups, num_channels)
    elif name == "layer":
        return nn.LayerNorm(num_channels)
    else:
        raise ValueError(f"Unsupported normalization: {name}")
    
def get_act(name: str, **kwargs):
    """
    获取指定类型的激活函数
    
    Args:
        name (str): 激活函数类型 ('relu', 'gelu', 'leaky_relu', 'swish', 'sigmoid')
        **kwargs: 其他参数（如 leaky_relu_slope)
    
    Returns:
        nn.Module: 对应的激活函数
    """
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == "leaky_relu":
        slope = kwargs.get("leaky_relu_slope", 0.1)
        return nn.LeakyReLU(negative_slope=slope)
    elif name == "swish":
        return Swish()
    elif name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {name}")