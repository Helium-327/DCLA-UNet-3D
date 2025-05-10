# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/03/08 21:17:10
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 重写UNETR
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels=4, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 == nn.LayerNorm(embed_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim* mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) 
            for _ in range(num_layers)
            ])
        
    def forward(self, x, return_all=False):
        if return_all:
            features = []
            for layer in self.layers:
                x = layer(x)
                features.append(x)
            return features                 # 输出所有层的特征
        else:
            for layer in self.layers:
                x = layer(x)
            return x                        # 输出最后一层的特征

        
class DecoderStage(nn.Module):
    def __init__(self, in_dim, out_dim, num_upscale=4):
        super().__init__()
        layers = []
        current_dim = in_dim
        for i in range(num_upscale):
            layers.append(
                nn.ConvTranspose3d(
                    current_dim, current_dim//2, kernel_size=2, stride=2
                )
            )
            layers.append(nn.InstanceNorm3d(current_dim//2))
            layers.append(nn.ReLU(inplace=True))
            current_dim = current_dim // 2
        self.upsample = nn.Sequential(*layers)
        self.final_conv = nn.Conv3d(current_dim, out_dim, 1)
        
    def forward(self, x, spatial_shape):
        B, _, E = x.shape
        x = x.view(B, *spatial_shape, E).permute(0, 4, 1, 2, 3)
        x = self.upsample(x)
        x = self.final_conv(x)
        return x
    
class UNETR(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, 
                 patch_size=16, embed_dim=768,
                 num_layers=12, num_heads=12):
        super().__init__()
        self.patch_embedding = PatchEmbedding3D(in_channels, patch_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 128, embed_dim))
        self.dropout = nn.Dropout(0.1)
        
