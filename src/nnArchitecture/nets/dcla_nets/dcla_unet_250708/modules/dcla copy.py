import torch
import torch.nn as nn

from nnArchitecture.nets.dcla_nets.dcla_unet_250615.modules.grn import GRN3D
from .mambas import MambaLayer
from .commons import *


#FIXME: 尺寸变化错误
class MultiScaleDownsampler(nn.Module):
    """多尺度空间压缩器（无归一化版本）
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
    Args:
        in_size (int): 输入特征空间尺寸
        min_size (int): 最小下采样尺寸
        kernel_sizes (list): 不同分支的卷积核尺寸列表
    """
    def __init__(self, in_size=128, min_size=8, kernel_sizes=[3,5,7]):
        super().__init__()
        self.downsample_branches = nn.ModuleList([
            self._create_downsample_path(k, in_size, min_size) for k in kernel_sizes
        ])
        
    def _create_downsample_path(self, kernel_size, input_size, min_size):
        layers = []
        current_size = input_size
        while current_size > min_size:
            layers.append(nn.Sequential(
                nn.Conv3d(1, 1, kernel_size, stride=2, padding=kernel_size//2),
                nn.GELU(),
                nn.BatchNorm3d(1)
            ))
            current_size //= 2
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return sum(branch(x) for branch in self.downsample_branches)

class ChannelProjector(nn.Module):
    """通道维度投影器
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
    Args:
        in_channels (int): 输入通道数
        proj_channels (int): 投影通道数
        proj_kernel (int): 投影卷积核尺寸
    """
    def __init__(self, in_channels, proj_channels, proj_kernel=1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv3d(in_channels, proj_channels, proj_kernel, padding=proj_kernel//2),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.projector(x)
    
class ChannelsAttention(nn.Module):
    """通道注意力机制
                                                          
    Args:
        num_features (int): 输入特征数量
        fusion_kernel (int): 融合卷积核尺寸
    """
    def __init__(self, in_channels, out_channels, ratio=2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(in_channels, in_channels//ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(in_channels//ratio, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, features):
        return self.attn(features) * features
    
class SpatialAttention(nn.Module):
    """空间注意力机制
                                                          
    Args:
        num_features (int): 输入特征数量
        fusion_kernel (int): 融合卷积核尺寸
    """
    def __init__(self, fusion_kernel=3):
        super().__init__()
        self.spatial_fusion_attn = nn.Sequential(
            nn.Conv3d(2, 1, fusion_kernel, padding=fusion_kernel//2),
            nn.Sigmoid()
        )
    def forward(self, features):  
        mean, _ = torch.mean(features, dim=1, keepdim=True)
        max = torch.max(features, dim=1, keepdim=True)
        fused = torch.cat([mean, max], dim=1)
        return self.spatial_fusion_attn(fused) * features
    
class CrossLevelFeatureFuserv2(nn.Module):
    """跨层特征融合器
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
    Args:
        num_features (int): 输入特征数量
        fusion_kernel (int): 融合卷积核尺寸
    """
    def __init__(self, in_channels, split_ratio=2, fusion_kernel=3):
        super().__init__()
        self.split_ratio = split_ratio
        
        self.channel_attn = ChannelsAttention(in_channels, in_channels)
        self.spatial_attn = SpatialAttention(fusion_kernel=7)
        
        self.feature_fuser = nn.Sequential(
            nn.Conv3d(in_channels//split_ratio, in_channels//split_ratio, fusion_kernel, padding=fusion_kernel//2),
            nn.BatchNorm3d(in_channels),
            nn.GELU()
        )
        
    def forward(self, features):
        features = self.channel_attn(features)
        features = self.spatial_attn(features)
        y, feats = torch.chunk(features, self.split_ratio, dim=1)
        y = y * self.feature_fuser(feats)
        return y

class DynamicCrossAttentionNoNorm(nn.Module):
    """动态跨层注意力机制 (DCLA-NoNorm)
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
    Architecture:
        [Encoder Features] → ChannelProjector → MultiScaleDownsampler ↘
                                                              CrossLevelFeatureFuser → AttnMap
    Args:
        ch_list (list): 各层输入通道数列表
        feat_sizes (list): 各层特征空间尺寸列表
        min_size (int): 最小下采样尺寸
    """
    def __init__(self, ch_list, proj_channels, feat_sizes, min_size=8):
        super().__init__()
        # 初始化三个核心组件
        self.channel_projectors = nn.ModuleList([
            ChannelProjector(ch, proj_channels) for ch in ch_list
        ])
        
        self.spatial_compressors = nn.ModuleList([
            MultiScaleDownsampler(size, min_size) for size in feat_sizes
        ])
        
        self.feature_fuser = CrossLevelFeatureFuserv2(len(ch_list), split_ratio=2)
        
    def forward(self, encoder_features, x):
        # 特征处理流程
        processed = []
        for feat, proj, compressor in zip(encoder_features, 
                                        self.channel_projectors,
                                        self.spatial_compressors):
            # 通道投影 → 空间压缩
            compressed = compressor(proj(feat))
            processed.append(compressed.squeeze(1))
            
        # 生成注意力图
        attn_map = self.feature_fuser(torch.stack(processed, dim=1))
        return attn_map * x
    
class AdaptiveSpatialCondenser(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 kernel_size=[7,5,3], 
                 in_size=128, 
                 min_size=8,
                 fusion_mode='concat',
                 norm_type = "batch",
                 act_type = "gelu",
                 ):
        super().__init__()
        self.norm_type = norm_type
        self.act_type = act_type
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
                get_norm(self.norm_type, self.out_channels, instance_norm_affine=True),
                get_act(self.act_type)
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
    
class DynamicCrossLevelAttentionv2(nn.Module): #MSFA
    def __init__(self, 
                 ch_list, 
                 feats_size, 
                 min_size=8, 
                 mamba_dim=16, 
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
                    nn.BatchNorm3d(1),
                    nn.ReLU6()
                    ))
            
        for feat_size in feats_size:
            self.down_layers.append(
                AdaptiveSpatialCondenser(
                    in_channels=1, 
                    out_channels=1, 
                    kernel_size=self.kernel_size, 
                    in_size=feat_size, 
                    min_size=8,
                    fusion_mode=self.fusion_mode,  # 'concat' or 'add'，
                    norm_type = "batch",
                    act_type = "relu"
                    )
                )
            
        self.conv = nn.Sequential(
            nn.Conv3d(len(self.kernel_size),
                      1, 
                      kernel_size=1, 
                      padding=0
                      ),
            nn.BatchNorm3d(1),
            nn.ReLU6(),
            nn.Conv3d(1, 1, kernel_size=1, padding=0)
        )
        
        self.fusion = nn.Sequential(
            GRN3D(len(self.ch_list)),
            nn.Conv3d(len(self.ch_list), len(self.ch_list), kernel_size=3, padding=3//2, groups=len(self.ch_list)),
            nn.Conv3d(len(self.ch_list), len(self.ch_list), kernel_size=1),
            nn.BatchNorm3d(len(self.ch_list)),
            nn.ReLU6()
        )
        
        self.attn = nn.Sequential(
            nn.Conv3d(len(self.ch_list), 1, kernel_size=fusion_kernel, padding=fusion_kernel//2),
            nn.Sigmoid()
        )

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
        fusion = self.fusion(torch.stack(downs, dim=1))
        attn = self.attn(fusion)
        out = attn * x
        return out
    

class DynamicCrossLevelAttentionv3(nn.Module): #MSFA
    def __init__(self, 
                 ch_list, 
                 feats_size, 
                 min_size=8, 
                 mamba_dim=16, 
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
        
        self.mamba = MambaLayer(
            dim=len(ch_list),  # 输入通道数
            d_state=mamba_dim,
            d_conv=4,
            expand=2,
            channel_token=False
        )
        
        self.proj = nn.Conv3d(len(self.ch_list), len(self.ch_list), kernel_size=3, padding=1, groups=len(self.ch_list))
        
        self.attn = nn.Sequential(
            nn.Conv3d(len(self.ch_list), 1, kernel_size=fusion_kernel, padding=fusion_kernel//2),
            nn.Sigmoid()
        )
        
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
        fused = torch.stack(downs, dim=1)
        fused = self.mamba(fused + self.proj(fused)) 
        attn = self.attn(fused)        
        out = attn * x
        return out
        