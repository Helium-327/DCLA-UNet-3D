from .commons import  *
from .slk import ResDWAConvBlock
from .msf import MutilScaleFusionBlock


class AxioFusionBlock(nn.Module):  # Axial-Dynamic Multi-scale Fusion Block with Residual Learning
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 dilations=[1,2,3],
                 norm_type="batch",
                 act_type="gelu",
                 dropout_rate=0.1
                ):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            get_norm(norm_type, out_channels),
            get_act(act_type)
        )
        
        self.resblock = ResDWAConvBlock(out_channels, 
                                        out_channels, 
                                        kernel_size=kernel_size, 
                                        norm_type=norm_type, 
                                        act_type=act_type
                                        )
        self.ms_block = MutilScaleFusionBlock(out_channels,
                                              out_channels,
                                              kernel_size=3,
                                              dilations=dilations,
                                              norm_type=norm_type,
                                              act_type=act_type
                                            )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1),
            get_norm(norm_type, out_channels),
            get_act(act_type)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        # map = self.avg(out)
        out = self.resblock(out)
        out = self.ms_block(out)
        # out = map + out
        out = self.conv2(out)
        return out
