import torch
import torch.nn as nn

# from nnArchitecture.nets.dcla_nets.dcla_unet_250615.modules.grn import GRN3D
from .commons import get_norm, get_act


class SingleBranchDownsampler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, in_size, min_size, norm_type="batch", act_type="gelu"):
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_size = in_size
        
        norm_layer = get_norm(norm_type, out_channels) if norm_type != "instance" else get_norm(norm_type, out_channels, instance_norm_affine=True)
        while current_size > min_size:
            self.layers.append(nn.Sequential(
                nn.Conv3d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    stride=2, 
                    padding=kernel_size//2
                    ),
                norm_layer,
                get_act(act_type)
            ))
            current_size //= 2

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class MultiBranchController(nn.Module):
    """多分支控制器
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_sizes (list): 卷积核尺寸列表
        in_size (int): 输入特征空间尺寸
        min_size (int): 目标最小尺寸
        norm_type (str): 归一化类型
        act_type (str): 激活函数类型
    """
    def __init__(self, in_channels, out_channels, kernel_sizes, in_size, min_size, norm_type, act_type):
        super().__init__()
        self.branches = nn.ModuleList([
            SingleBranchDownsampler(
                in_channels, 
                out_channels, 
                k, 
                in_size, 
                min_size, 
                norm_type, 
                act_type
            ) for k in kernel_sizes
        ])
    
    def forward(self, x):
        return [branch(x) for branch in self.branches]  

  
class FeatureFuser(nn.Module):
    """多分支特征融合器
    Args:
        fusion_mode (str): 融合模式 ('concat' 或 'add')
        num_branches (int): 分支数量
    """
    def __init__(self, fusion_mode, num_branches):
        super().__init__()
        self.fusion_mode = fusion_mode
        if fusion_mode == 'concat':
            self.channel_reducer = nn.Conv3d(num_branches, 1, 1)
    
    def forward(self, branch_outputs):
        if self.fusion_mode == 'concat':
            return self.channel_reducer(torch.cat(branch_outputs, dim=1))
        elif self.fusion_mode == 'add':
            return torch.sum(torch.stack(branch_outputs), dim=0)
        else:
            raise ValueError("Invalid fusion mode. Choose from 'concat' or 'add'.")      


class AdaptiveSpatialCondenser(nn.Module):
    """自适应空间压缩器 (集成模块)
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (list): 卷积核尺寸列表
        in_size (int): 输入特征空间尺寸
        min_size (int): 目标最小尺寸
        fusion_mode (str): 融合模式 ('concat' 或 'add')
        norm_type (str): 归一化类型
        act_type (str): 激活函数类型
    """
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 kernel_size=[3,5,7], 
                 in_size=128, 
                 min_size=8,
                 fusion_mode='concat',
                 norm_type="batch",
                 act_type="relu"):
        super().__init__()
        self.branch_controller = MultiBranchController(
            in_channels, 
            out_channels, 
            kernel_size, 
            in_size, 
            min_size, 
            norm_type, 
            act_type
        )
        self.fuser = FeatureFuser(fusion_mode, len(kernel_size))
    
    def forward(self, x):
        branch_outs = self.branch_controller(x)
        return self.fuser(branch_outs)

class ChannelCompressor(nn.Module):
    """通道维度压缩器
    Args:
        in_channels (int): 输入通道数
        squeeze_kernel (int): 压缩卷积核尺寸
    """
    def __init__(self, in_channels, squeeze_kernel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, 1, squeeze_kernel, padding=squeeze_kernel//2),
            nn.BatchNorm3d(1),
            nn.ReLU6()
        )
    
    def forward(self, x):
        return self.layer(x)

class FeatureAligner(nn.Module):
    """特征空间对齐器
    Args:
        in_size (int): 输入特征空间尺寸
        min_size (int): 目标最小尺寸
        kernel_sizes (list): 卷积核尺寸列表
        fusion_mode (str): 融合模式 ('concat' 或 'add')
    """
    def __init__(self, in_size, min_size, kernel_sizes, fusion_mode):
        super().__init__()
        self.compressor = AdaptiveSpatialCondenser(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_sizes,
            in_size=in_size,
            min_size=min_size,
            fusion_mode=fusion_mode,
            norm_type="batch",
            act_type="relu"
        )
        self.target_size = min_size
        self.fusion_mode = fusion_mode
        
        # 当fusion_mode='concat'时需初始化通道压缩卷积
        if fusion_mode == 'concat':
            self.channel_reducer = nn.Sequential(
                nn.Conv3d(len(kernel_sizes), 1, kernel_size=1),
                nn.BatchNorm3d(1),
                nn.ReLU6()
            )
    
    def forward(self, x):
        if x.size(2) != self.target_size:
            compressed = self.compressor(x)
            if self.fusion_mode == 'concat':
                return self.channel_reducer(compressed)
            return compressed
        return x

class CrossLevelFuser(nn.Module):
    """跨层级特征融合器
    Args:
        num_levels (int): 特征层级数量
    """
    def __init__(self, num_levels):
        super().__init__()
        self.grn = GRN3D(num_levels)
        self.depthwise = nn.Conv3d(num_levels, num_levels, 3, padding=1, groups=num_levels)
        self.pointwise = nn.Conv3d(num_levels, num_levels, 1)
        self.norm_act = nn.Sequential(
            nn.BatchNorm3d(num_levels), 
            nn.ReLU6()
        )
    
    def forward(self, x):
        x = self.grn(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.norm_act(x)


class AttentionGenerator(nn.Module):
    """空间注意力生成器
    Args:
        in_channels (int): 输入通道数
        fusion_kernel (int): 融合卷积核尺寸
    """
    def __init__(self, in_channels, fusion_kernel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, 1, fusion_kernel, padding=fusion_kernel//2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layer(x)


class GRN3D(nn.Module):
    """
    Global Response Normalization for 3D ConvNets (ConvNeXt V2 style)
    适配输入形状: [B, C, D, H, W]
    """
    def __init__(self, channels, eps=1e-6):
        """
        Args:
            channels (int): 输入特征图的通道数
            eps (float): 防止除零的小常数
        """
        super().__init__()
        self.eps = eps
        # 通道缩放参数（gamma）和偏置（beta）
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        
        # 特征转换层（两层1x1x1卷积）
        self.fc1 = nn.Conv3d(channels, channels, 1)
        self.fc2 = nn.Conv3d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        """
        前向传播
        Args:
            x (Tensor): 输入特征图，形状 [B, C, D, H, W]
        Returns:
            Tensor: GRN处理后的特征图，形状不变
        """
        # 1. 全局特征聚合
        gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)  # L2范数聚合空间信息 [B, C, 1, 1, 1]
        
        # 2. 特征校准
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)   # 通道归一化
        
        # 3. 特征转换
        z = self.fc1(nx)
        z = self.act(z)
        z = self.fc2(z)
        
        # 4. 门控机制
        return x * (self.gamma * z.sigmoid() + self.beta + 1) 

class DynamicCrossLevelAttentionv2(nn.Module):
    """动态跨层级注意力机制 v2 (集成模块)
    Args:
        ch_list (list): 各层输入通道数列表
        feats_size (list): 各层特征空间尺寸列表
        min_size (int): 目标最小尺寸
        squeeze_kernel (int): 通道压缩核尺寸
        down_kernel (list): 下采样卷积核尺寸列表
        fusion_kernel (int): 注意力融合核尺寸
        fusion_mode (str): 特征融合模式 ('concat' 或 'add')
    """
    def __init__(self, 
                 ch_list, 
                 feats_size, 
                 min_size=8, 
                 squeeze_kernel=1,
                 down_kernel=[3,5,7], 
                 fusion_kernel=1,
                 fusion_mode='add'):
        super().__init__()
        self.min_size = min_size
        
        # 初始化子模块
        self.channel_compressors = nn.ModuleList([
            ChannelCompressor(ch, squeeze_kernel) for ch in ch_list
        ])
        
        self.spatial_aligners = nn.ModuleList([
            FeatureAligner(size, min_size, down_kernel, fusion_mode)
            for size in feats_size
        ])
        
        self.fuser = CrossLevelFuser(len(ch_list))
        self.attn_gen = AttentionGenerator(len(ch_list), fusion_kernel)

    def forward(self, encoder_feats, x):
        aligned_feats = []
        
        # 通道压缩 → 空间对齐
        for feat, compressor, aligner in zip(encoder_feats, 
                                           self.channel_compressors, 
                                           self.spatial_aligners):
            squeezed = compressor(feat)
            aligned = aligner(squeezed)
            aligned_feats.append(aligned.squeeze(1))
        
        # 跨层融合 → 生成注意力 → 加权输出
        stacked = torch.stack(aligned_feats, dim=1)
        fused = self.fuser(stacked)
        attn = self.attn_gen(fused)
        return attn * x
        
        