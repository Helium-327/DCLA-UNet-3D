# 定义公共参数
BASE_ARGS = {"in_channels":4, "out_channels":4}

from nnArchitecture.nets import *

model_register = {
    
    "SOTA Models": {
        "UNet3D":       UNet3D(**BASE_ARGS),
        "AttUNet3D":    AttUNet3D(**BASE_ARGS),
        "UNETR":        UNETR(**BASE_ARGS, 
                       img_size=(128, 128, 128),
                       feature_size=16,
                       dropout_rate=0.2,
                       norm_name="instance",
                       spatial_dims=3
                       ),
        "UNETR_PP":     UNETR_PP(**BASE_ARGS, 
                            feature_size=16,
                            hidden_size=256,
                            num_heads=8,
                            pos_embed="perceptron",
                            norm_name="instance",
                            dropout_rate=0.1,
                            depths=[3, 3, 3, 3],
                            dims=[32, 64, 128, 256],
                            conv_op=nn.Conv3d,
                            do_ds=False,
                        ),
        "SegFormer3D":  SegFormer3D(**BASE_ARGS),
        "Mamba3d":      Mamba3d(**BASE_ARGS),
        "MogaNet":      MogaNet(**BASE_ARGS),
    },
    
    "Baseline Models": {
        "DWResUNet":                            DWResUNet(**BASE_ARGS),
        "ResUNetBaseline_S":                    ResUNetBaseline_S(**BASE_ARGS),
        "ResUNetBaseline_M":                    ResUNetBaseline_M(**BASE_ARGS),
    },
    
    "DCLA_UNet final": {
        "SLK_UNet_final":                       SLK_UNet_final(**BASE_ARGS),
        "MSF_UNet_final":                       MSF_UNet_final(**BASE_ARGS),
        "SLK_MSF_UNet_final":                   SLK_MSF_UNet_final(**BASE_ARGS),
        "DCLA_UNet_final":                      DCLA_UNet_final(**BASE_ARGS)
    },
    
    
}