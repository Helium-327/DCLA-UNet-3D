from turtle import forward
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from nnArchitecture.commons import (
    init_weights_3d,
    UpSample
)

from nnArchitecture.nets.dcla_nets.dcla_unet_250606.mm import *
from utils.test_unet import test_unet

class UnifiedUNet(nn.Module):
    """统一消融实验架构，支持以下模块配置：
    - 编码器类型：['res', 'slk', 'msf']
    - 解码器类型：['basic', 'msf', 'attention']
    - 是否启用DCLA
    - 是否启用SE注意力
    - 是否启用残差连接
    """

    ARCH_CONFIG = {
        'baseline': {
            'encoder_type': 'res',
            'decoder_type': 'basic',
            'use_dcla': False,
            'use_se': False,
            'use_residual': True
        },
        'slk': {
            'encoder_type': 'slk',
            'decoder_type': 'basic', 
            'use_dcla': False,
            'use_se': True,
            'use_residual': True
        },
        'dcla': {
            'encoder_type': 'slk',
            'decoder_type': 'msf',
            'use_dcla': True,
            'use_se': True,
            'use_residual': True
        }
    }

    def __init__(self, 
                 in_channels=4,
                 out_channels=4,
                 kernel_size=7,
                 f_list=[32, 64, 128, 256],
                 trilinear=True,
                 se_ratio=16,
                 config_name='baseline',
                 custom_config=None):
        """
        Args:
            config_name: 预定义配置名称，可选['baseline', 'slk', 'dcla']
            custom_config: 自定义配置字典，会覆盖预定义配置
        """
        super().__init__()
        
        # 合并配置
        self.config = self.ARCH_CONFIG[config_name].copy()
        if custom_config:
            self.config.update(custom_config)
            
        # 生成模型标识
        self._generate_model_name()
        
        # 初始化共享组件
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self._init_encoder(in_channels, f_list, kernel_size, se_ratio)
        self._init_decoder(f_list, kernel_size)
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
        # DCLA模块
        if self.config['use_dcla']:
            self.dcla = DynamicCrossLevelAttention(
                ch_list=f_list, 
                feats_size=[128, 64, 32, 16],
                min_size=8,
                squeeze_kernel=1,
                down_kernel=[7],
                fusion_kernel=1
            )
            
        self.apply(init_weights_3d)

    def _generate_model_name(self):
        """自动生成可读的模型名称"""
        name_parts = []
        
        # 编码器类型
        encoder_map = {'res': 'Res', 'slk': 'SLK', 'msf': 'MSF'}
        name_parts.append(encoder_map[self.config['encoder_type']])
        
        # 解码器类型
        decoder_map = {'basic': '', 'msf': '-MSF', 'attention': '-Attn'}
        name_parts.append(decoder_map[self.config['decoder_type']])
        
        # 特殊模块
        if self.config['use_dcla']: name_parts.append('-DCLA')
        if not self.config['use_se']: name_parts.append('-NoSE')
        if not self.config['use_residual']: name_parts.append('-NoRes')
        
        self.model_name = "UNet" + "".join(name_parts)

    def _init_encoder(self, in_ch, f_list, kernel_size, se_ratio):
        """初始化编码器"""
        encoder_blocks = []
        current_ch = in_ch
        
        for out_ch in f_list:
            encoder_blocks.append(
                self._build_encoder_block(
                    current_ch, out_ch, kernel_size, se_ratio
                )
            )
            current_ch = out_ch
            
        self.encoder = nn.Sequential(*encoder_blocks)

    def _build_encoder_block(self, in_ch, out_ch, kernel_size, se_ratio):
        """构建单个编码器块"""
        layers = []
        
        # 主干卷积
        if self.config['encoder_type'] == 'res':
            layers.append(ResConv3D_S_BN(in_ch, out_ch, kernel_size=3))
        elif self.config['encoder_type'] == 'slk':
            layers.append(SlimLargeKernelBlockv4(in_ch, out_ch, kernel_size))
        elif self.config['encoder_type'] == 'msf':
            layers.append(MutilScaleFusionBlock(in_ch, out_ch))
            
        # SE注意力
        if self.config['use_se']:
            layers.append(SqueezeExcitation(out_ch, se_ratio))
            
        # 残差连接
        if self.config['use_residual']:
            layers.append(nn.Identity())  # 实际实现需根据具体模块调整
            
        return nn.Sequential(*layers)

    def _init_decoder(self, f_list, kernel_size):
        """初始化解码器"""
        self.upsamples = nn.ModuleList([
            UpSample(ch, ch, True) for ch in reversed(f_list)
        ])
        
        self.decoder_blocks = nn.ModuleList([
            self._build_decoder_block(in_ch, out_ch, kernel_size)
            for in_ch, out_ch in zip(
                [f_list[3]*2, f_list[2]*2, f_list[1]*2, f_list[0]*2],
                [f_list[3]//2, f_list[2]//2, f_list[1]//2, f_list[0]]
            )
        ])

    def _build_decoder_block(self, in_ch, out_ch, kernel_size):
        """构建单个解码器块"""
        layers = []
        
        # 注意力机制
        if self.config['decoder_type'] == 'attention':
            layers.append(EfficientAttentionBlock(in_ch, spatial_kernal=7, ratio=16))
            
        # 多尺度融合
        if self.config['decoder_type'] == 'msf':
            layers.append(MutilScaleFusionBlock(in_ch, out_ch))
        else:
            layers.append(nn.Conv3d(in_ch, out_ch, 1))
            
        # 残差连接
        if self.config['use_residual']:
            layers.append(nn.Identity())  # 实际实现需根据具体模块调整
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        features = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i < len(self.encoder)-1:  # 保存前4层特征
                features.append(x)
                x = self.MaxPool(x)
                
        # DCLA
        if self.config['use_dcla'] and hasattr(self, 'dcla'):
            x = self.dcla(features, x) + x
            
        # Decoder
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            x = up(x)
            if self.config.get('use_skip', True) and i < len(features):
                x = torch.cat([x, features[-(i+1)]], dim=1)
            x = dec(x)
            
        return self.outc(x)

    def __repr__(self):
        return f"{self.model_name}(Params: {sum(p.numel() for p in self.parameters())/1e6:.2f}M)"

# -------------------- 预定义模型 --------------------
class ResUNetBaseline_S(UnifiedUNet):
    def __init__(self, **kwargs):
        super().__init__(config_name='baseline', **kwargs)

class SLK_UNet(UnifiedUNet):
    def __init__(self, **kwargs):
        super().__init__(config_name='slk', **kwargs)

class DCLA_UNet_250606(UnifiedUNet):
    def __init__(self, **kwargs):
        super().__init__(config_name='dcla', **kwargs)

# -------------------- 实验配置示例 --------------------
if __name__ == "__main__":
    # 基线模型
    baseline = ResUNetBaseline_S()
    print(baseline)  # 输出: UNetRes(Params: 3.23M)
    
    # 消融实验模型
    experiments = {
        'NoSE': UnifiedUNet(custom_config={'use_se': False}),
        'NoDCLA': UnifiedUNet(custom_config={'use_dcla': False}),
        'SLK-MSF': UnifiedUNet(custom_config={
            'encoder_type': 'slk',
            'decoder_type': 'msf'
        }),
        'FullModel': UnifiedUNet(custom_config={
            'encoder_type': 'slk',
            'decoder_type': 'attention',
            'use_dcla': True,
            'use_se': True
        })
    }
    
    # 测试所有变体
    for name, model in experiments.items():
        test_unet(model_class=model.__class__, batch_size=1)
        print(f"{name}: {model.model_name}")