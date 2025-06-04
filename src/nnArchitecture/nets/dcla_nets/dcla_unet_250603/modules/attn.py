import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .commons import Swish

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


class GatedAdaptiveSpatialCondenser(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 kernel_size=[7,5,3], 
                 in_size=128, 
                 min_size=8,
                 fusion_mode='gated'  # 新增门控模式
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.in_size = in_size
        self.min_size = min_size
        self.fusion_mode = fusion_mode
        
        # 动态门控权重生成器
        self.gate_conv = nn.Sequential(
            nn.Conv3d(len(kernel_size)*out_channels, len(kernel_size), 1),
            nn.InstanceNorm3d(len(kernel_size)),
            nn.GELU(),
            nn.Conv3d(len(kernel_size), len(kernel_size), 1),
            nn.Softmax(dim=1)
        ) if fusion_mode == 'gated' else None
        
        self.branches = self._build_multi_branchs()

    def _build_multi_branchs(self):
        return nn.ModuleList([self._build_single_branch(k) for k in self.kernel])

    def _build_single_branch(self, kernel_size):
        layers = []
        current_size = self.in_size
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
        branch_outputs = [branch(x) for branch in self.branches]
        
        # 门控融合实现
        if self.fusion_mode == 'gated':
            concat_features = torch.cat(branch_outputs, dim=1)
            gate_weights = self.gate_conv(concat_features)
            weighted_features = [w * f for w, f in zip(
                torch.unbind(gate_weights, dim=1), 
                branch_outputs
            )]
            return torch.sum(torch.stack(weighted_features), dim=0)
            
        elif self.fusion_mode == 'concat':
            return torch.cat(branch_outputs, dim=1)
        elif self.fusion_mode == 'add':
            return torch.sum(torch.stack(branch_outputs), dim=0)
        else:
            raise ValueError(f"Invalid fusion mode: {self.fusion_mode}")

class DynamicCrossLevelAttention(nn.Module): #MSFA
    def __init__(self, 
                 ch_list, 
                 feats_size, 
                 min_size=8, 
                 squeeze_kernel=1,
                 down_kernel=[7], 
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
                    nn.Conv3d(in_channels=ch, out_channels=1, kernel_size=squeeze_kernel, padding=squeeze_kernel//2),
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
                    min_size=self.min_size,
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
    

class HybridDASPP(nn.Module):
    def __init__(self, in_channels, dilation_rates=[1,2,3,6]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 3, 
                        padding=d, dilation=d),
                nn.InstanceNorm3d(in_channels),
                nn.GELU()
            ) for d in dilation_rates
        ])
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels*len(dilation_rates), in_channels, 1),
            SwishECA(in_channels, gamma=1, b=2)
        )
        
    def forward(self, x):
        branch_features = [branch(x) for branch in self.branches]
        return self.fusion(torch.cat(branch_features, dim=1))
    
    
class DynamicFeatureCalibration(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dynamic_conv = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            Swish(),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _, _ = x.shape
        weights = self.global_pool(x).view(b, c)
        dynamic_weights = self.dynamic_conv(weights).view(b, c, 1, 1, 1)
        return x * dynamic_weights
    
class EnhancedCrossLevelAttention(nn.Module):
    def __init__(self, ch_list, feats_size, min_size=8, fusion_mode='gated'):
        super().__init__()
        # 初始化原有组件
        self.min_size = min_size
        self.squeeze_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, 1, 1),
                nn.InstanceNorm3d(1),
                HybridDASPP(1, dilation_rates=[1,2,3])  # 新增多尺度处理
            ) for ch in ch_list
        ])
        
        self.down_layers = nn.ModuleList([
            GatedAdaptiveSpatialCondenser(
                in_channels=1,
                out_channels=1,
                kernel_size=[3,5,7],  # 扩展卷积核范围
                in_size=feat_size,
                min_size=self.min_size,
                fusion_mode='gated'  # 新增门控融合
            ) for feat_size in feats_size
        ])
        
        # 新增跨尺度交互模块
        self.cross_scale_attn = LightweightSpatialAttention3D(kernel_size=7)
        
        # 动态特征校准
        self.dfc = DynamicFeatureCalibration(len(ch_list))
        
    def forward(self, encoder_feats, x):
        # 多尺度特征提取
        squeezed = [squeeze(feat) for squeeze, feat in zip(self.squeeze_layers, encoder_feats)]
        
        # 多级下采样与校准
        downs = []
        for feat, down_layer in zip(squeezed, self.down_layers):
            down_feat = down_layer(feat)
            # 跨尺度注意力
            # down_feat = self.cross_scale_attn[0](down_feat) * down_feat
            down_feat = self.cross_scale_attn(down_feat) * down_feat
            downs.append(down_feat)
            
        # 动态特征融合
        fused = self.dfc(torch.cat(downs, dim=1))
        attn = torch.sigmoid(fused.sum(dim=1, keepdim=True))
        
        return attn * x + x  # 残差连接