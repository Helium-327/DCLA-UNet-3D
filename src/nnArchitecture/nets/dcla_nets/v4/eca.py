import torch
import math
from torch import nn




class ECA3D(nn.Module):
    """3D版本ECA模块，适配体积数据"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 3D全局池化 [D,H,W] -> [1,1,1]
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 动态计算1D卷积核大小 (保持原论文公式)
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            Swish(),
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)  + self.max_pool(x).view(b, c)        # [B,C]
        y = y.unsqueeze(1)                       # [B,1,C]
        y = self.conv(y)                         # 跨通道交互
        y = self.sigmoid(y).view(b, c, 1, 1, 1)  # [B,C,1,1,1]
        return x * y.expand_as(x)                # 3D广播相乘
    
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
    
class ECA3Dv2(nn.Module):
    """3D版本ECA模块，适配体积数据"""
    def __init__(self, in_channels, out_channels, gamma=2, b=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 3D全局池化 [D,H,W] -> [1,1,1]
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 动态计算1D卷积核大小 (保持原论文公式)
        kernel_size = int(abs((math.log2(in_channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.attn = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                nn.ReLU(),
                nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.Sigmoid()
        ) 

    def forward(self, encoder_feats, x):
        b, c, _, _, _ = x.shape
        feats = [self.avg_pool(feat).view(b, -1) + self.max_pool(feat).view(b, -1) for feat in encoder_feats]
        y = torch.cat(feats, dim=1)
        y = y.unsqueeze(1)                       # [B,1,C]
        attn = self.attn(y).view(b, -1, 1, 1, 1)  # 跨通道交互 [B,C,1,1,1]
        return x * attn              # 3D广播相乘
    
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(x)