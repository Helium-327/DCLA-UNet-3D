# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/05/05 11:39:46
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v8.0
*      FEATURES: 
*      STATE:     
*      CHANGE ON: 
=================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# ✨ 创新点1： MADAG 模块 —— 空间维度分离的注意力模块
# ✨ 创新点2： 动态维度加权融合 - 动态调整平衡深度、高度和宽度维度的重要性
# ✨ 创新点3： 动态维度权重参数 - 允许模型在不同维度上学习不同的权重
# ✨ 创新点4： 自适应平均池化 - 用于通道注意力模块

# 1. 使用GroupNorm代替InstanceNorm保留小目标特征
# 2. 采用坐标注意力保持空间关联性
# 3. 移除过度压缩的通道瓶颈
# 4. 引入可学习的温度系数动态调节注意力强度
# 5. 三维卷积分别处理不同维度，避免softmax压制


# @1: 使用GroupNorm 更好平衡通道独立性与相关性
class MultiAxisDualAttnGatev0(nn.Module):
    """
    多轴双注意力门机制
    参数:
        F_g (int): 门控信号的通道数
        F_l (int): 跳跃连接的通道数
        F_inter (int): 中间特征维度，需能被4整除
    """
    def __init__(self, F_g, F_l, F_inter):
        
        assert F_inter % 4 == 0, "F_inter必须能被4整除"
        super().__init__()
        # 特征转换层
        # 上采样路径
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_inter, kernel_size=1),
            nn.BatchNorm3d(F_inter)  # 更好平衡通道独立性与相关性
        )
        
        # 跳跃连接路径（可考虑去掉，保留跳跃连接的特征）
        self.W_x = nn.Sequential(                       
            nn.Conv3d(F_l, F_inter, kernel_size=1),
            nn.BatchNorm3d(F_inter)
        )
        
        # 空间注意力 - 使用分组卷积提高效率
        self.spatial_att = AxialPoolAttn3D(F_inter, F_l)  # 输出形状 [B,F_l,D,H,W]
        
        # 通道注意力 - 使用SE模块风格实现
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(F_inter, F_inter//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(F_inter//4, F_l, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化权重
    def forward(self, g, x):
        # 特征融合 # [B, F_inter, D, H, W]
        fused = self.relu(self.W_g(g) + self.W_x(x)) 
        # 空间注意力 - 一次性计算三个维度 
        spatial_map = self.spatial_att(fused)  # [B, F_l, D, H, W]

        # 通道敏感的充标定
        channel_map = self.channel_att(fused)  # [B, F_l,1, 1, 1]

        return x * spatial_map * channel_map 

class MultiAxisDualAttnGatev1(nn.Module):
    """
    多轴双注意力门机制
    参数:
        F_g (int): 门控信号的通道数
        F_l (int): 跳跃连接的通道数
        F_inter (int): 中间特征维度，需能被4整除
    """
    def __init__(self, F_g, F_l, F_inter):
        
        assert F_inter % 4 == 0, "F_inter必须能被4整除"
        super().__init__()
        # 特征转换层
        # 上采样路径
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_inter, kernel_size=1),
            nn.BatchNorm3d(F_inter)  # 更好平衡通道独立性与相关性
        )
        
        # 跳跃连接路径（可考虑去掉，保留跳跃连接的特征）
        self.W_x = nn.Sequential(                       
            nn.Conv3d(F_l, F_inter, kernel_size=1),
            nn.BatchNorm3d(F_inter)
        )
        
        # 空间注意力 - 使用分组卷积提高效率
        self.spatial_att = nn.Sequential(
            nn.Conv3d(F_inter, 1, kernel_size=1),
            nn.Sigmoid()
        )   # 输出形状 [B,1,D,H,W]
        
        # 通道注意力 - 使用SE模块风格实现
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(F_inter, F_inter//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(F_inter//4, F_l, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化权重
    def forward(self, g, x):
        # 特征融合 # [B, F_inter, D, H, W]
        fused = self.relu(self.W_g(g) + self.W_x(x)) 
        # 空间注意力 - 一次性计算三个维度 
        spatial_map = self.spatial_att(fused)  # [B, F_l, D, H, W]

        # 通道敏感的充标定
        channel_map = self.channel_att(fused)  # [B, F_l,1, 1, 1]

        return x * spatial_map * channel_map 

class MultiAxisDualAttnGate(nn.Module):
    """
    多轴双注意力门机制
    参数:
        F_g (int): 门控信号的通道数
        F_l (int): 跳跃连接的通道数
        F_inter (int): 中间特征维度，需能被4整除
    """
    def __init__(self, F_g, F_l, F_inter):
        
        assert F_inter % 4 == 0, "F_inter必须能被4整除"
        super().__init__()
        # 特征转换层
        # 上采样路径
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_inter, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=F_inter)  # 更好平衡通道独立性与相关性
        )
        
        # 跳跃连接路径（可考虑去掉，保留跳跃连接的特征）
        self.W_x = nn.Sequential(                       
            nn.Conv3d(F_l, F_inter, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=F_inter)
        )
        
        # 空间注意力 - 使用分组卷积提高效率
        self.spatial_att = CoordAttn3D(F_inter, F_l)  # 输出形状 [B,F_l,D,H,W]
        
        # 通道注意力 - 使用SE模块风格实现
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(F_inter, F_inter//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(F_inter//2, F_l, 1),
            nn.Sigmoid()
        )
        self.tau = nn.Parameter(torch.ones(1)*0.5)
        
        # 初始化权重
    def forward(self, g, x):
        # 特征融合 # [B, F_inter, D, H, W]
        fused = F.relu(self.W_g(g) + self.W_x(x)) 
        
        # 空间注意力 - 一次性计算三个维度 
        spatial_map = self.spatial_att(fused)  # [B, F_l, D, H, W]

        # 通道敏感的充标定
        # fused_mean = fused.mean(dim=(2, 3, 4), keepdim=True)
        channel_map = self.channel_att(fused)  # [B, F_l,1, 1, 1]
        
        tau = torch.sigmoid(self.tau)
        combined_weights = tau * spatial_map + (1 - tau) * channel_map
        return x * combined_weights
    

# @1: 添加归一化
# @2: 添加共享卷积层

class CoordAttn3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.base_conv = nn.Sequential(                     # @2
            nn.Conv3d(in_channels, in_channels//4, 1),  # 通道压缩
            nn.GroupNorm(4, in_channels//4),
            nn.ReLU(inplace=True)
        )
        self.d_conv = nn.Conv3d(in_channels//4, out_channels, (3, 1, 1), padding=(1, 0, 0))
        self.h_conv = nn.Conv3d(in_channels//4, out_channels, (1, 3, 1), padding=(0, 1, 0))
        self.w_conv = nn.Conv3d(in_channels//4, out_channels, (1, 1, 3), padding=(0, 0, 1))     
        
        self.dim_weights = nn.Parameter(torch.ones(3))   
        self.norm = nn.GroupNorm(4, out_channels)  #@1
        
    def forward(self, x):
        x = self.base_conv(x)
        d_feat = torch.sigmoid(self.norm(self.d_conv(x)))
        h_feat = torch.sigmoid(self.norm(self.h_conv(x)))
        w_feat = torch.sigmoid(self.norm(self.w_conv(x)))
        
        weights = F.softmax(self.dim_weights, dim=0)
        combined = weights[0] * d_feat + weights[1] * h_feat + weights[2] * w_feat
        return torch.sigmoid(self.norm(combined))
    
    
class AxialPoolAttn3D(nn.Module):
    """
    轴向池化注意力模块（显存优化版）
    特点：
    1. 使用轴向池化降低计算复杂度
    2. 并行分支结构保持各维度信息
    3. 轻量级注意力机制
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.d_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),  # 压缩D维度
            nn.Conv3d(in_channels, in_channels//2, 1),
            nn.ReLU(inplace=True)
        )
        
        self.h_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, None)),  # 压缩H维度
            nn.Conv3d(in_channels, in_channels//2, 1),
            nn.ReLU(inplace=True)
        )
        
        self.w_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, None, 1)),  # 压缩W维度
            nn.Conv3d(in_channels, in_channels//2, 1),
            nn.ReLU(inplace=True)
        )
        
        self.attn_gen = nn.Sequential(
            nn.Conv3d(in_channels//2 * 3, out_channels, 1),  # 合并三个分支
            nn.BatchNorm3d(out_channels),
            nn.Sigmoid()
        )
        
        # self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.res_scale = nn.Parameter(torch.zeros(1))
        self.res_conv = nn.Conv3d(in_channels, out_channels, 1)
        
    def forward(self, x):
        residual = self.res_conv(x)
        # x = self.channel_compress(x)
        
        # 各轴向池化
        d_feat = self.d_pool(x)  # [B, C//16, D, 1, 1]
        h_feat = self.h_pool(x)  # [B, C//16, 1, H, 1]
        w_feat = self.w_pool(x)  # [B, C//16, 1, 1, W]
        
        # 特征融合与注意力生成
        combined = torch.cat([
            d_feat.expand(-1,-1,x.size(2),-1,-1),
            h_feat.expand(-1,-1,-1,x.size(3),-1),
            w_feat.expand(-1,-1,-1, -1,x.size(4))
        ], dim=1)
        
        attn_map = self.attn_gen(combined)
        
        # 残差连接
        return residual * (attn_map + self.res_scale)