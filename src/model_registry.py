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
        "BaseLine_S_SLK_final":                 BaseLine_S_SLK_final(**BASE_ARGS),
        "BaseLine_S_DCLA_SLK_final":            BaseLine_S_DCLA_SLK_final(**BASE_ARGS),
        "BaseLine_S_MSF_final":                 BaseLine_S_MSF_final(**BASE_ARGS),
        "BaseLine_S_DCLA_MSF_final":            BaseLine_S_DCLA_MSF_final(**BASE_ARGS),
        "BaseLine_S_SLK_MSF_final":             BaseLine_S_SLK_MSF_final(**BASE_ARGS),
        "BaseLine_S_DCLA_final":                BaseLine_S_DCLA_final(**BASE_ARGS),
        "DCLA_UNet_final":                      DCLA_UNet_final(**BASE_ARGS),
        "DCLA_UNet_finalv2":                    DCLA_UNet_finalv2(**BASE_ARGS),
        "DCLA_UNet_finalv3":                    DCLA_UNet_finalv3(**BASE_ARGS),
    },
    
    "DCLA_UNet 250607": {
        "ResUNeXt_250607":                               ResUNeXt_250607(**BASE_ARGS),
        "ResUNeXt_DCLA_250607":                          ResUNeXt_DCLA_250607(**BASE_ARGS),
        
        "SLK_UNet_250607":                               SLK_UNet_250607(**BASE_ARGS),
        "SLK_SE_UNet_250607":                            SLK_SE_UNet_250607(**BASE_ARGS),
        
        "MSF_UNet_250607":                               MSF_UNet_250607(**BASE_ARGS),
        
        "SLK_MSF_UNet_250607":                           SLK_MSF_UNet_250607(**BASE_ARGS),
        "SLK_MSF_DCLA_UNet_250607":                      SLK_MSF_DCLA_UNet_250607(**BASE_ARGS), 
        "SLK_MSF_DCLA_NoRes_UNet_250607":                SLK_MSF_DCLA_NoRes_UNet_250607(**BASE_ARGS),
        "SLK_MSF_DCLA_k7_NoRes_UNet_250607":             SLK_MSF_DCLA_k7_NoRes_UNet_250607(**BASE_ARGS),
        "SLK_MSF_DCLAv2_NoRes_UNet_250607":              SLK_MSF_DCLAv2_NoRes_UNet_250607(**BASE_ARGS),
        "SLK_MSF_DCLAv3_NoRes_UNet_250607":              SLK_MSF_DCLAv3_NoRes_UNet_250607(**BASE_ARGS),
        "SLK_MSF_SEdown_UNet_250607":                    SLK_MSF_SEdown_UNet_250607(**BASE_ARGS),
        
        "SLK_MSF_SE_UNet_250607":                        SLK_MSF_SE_UNet_250607(**BASE_ARGS),
        "SLK_MSF_SE_DCLA_UNet_250607":                   SLK_MSF_SE_DCLA_UNet_250607(**BASE_ARGS),
        
        "DCLA_UNet_250607":                              DCLA_UNet_250607(**BASE_ARGS),
        "DCLA_UNet_NoRes_250607":                        DCLA_UNet_NoRes_250607(**BASE_ARGS),
        "AxFB_UNet_250607":                              AxFB_UNet_250607(**BASE_ARGS),
      
    },
    
    "DCLA_UNet 250615": {
        "DCLA_UNet_NoRes_250615":                         DCLA_UNet_NoRes_250615(**BASE_ARGS),
    },
    
    "DCLA_UNet 250627": {
        "SLK_UNet_250627":                               SLK_UNet_250627(**BASE_ARGS),
        
        "MSF_UNet_250627":                               MSF_UNet_250627(**BASE_ARGS),
        
        "SLK_MSF_UNet_250627":                           SLK_MSF_UNet_250627(**BASE_ARGS),
        
        "DCLA_UNet_NoRes_250627":                        DCLA_UNet_NoRes_250627(**BASE_ARGS)
      
    },
    "DCLA_UNet 250705": {
        "SLK_UNet_250705":                               SLK_UNet_250705(**BASE_ARGS),
        
        "MSF_UNet_250705":                               MSF_UNet_250705(**BASE_ARGS),
        
        "SLK_MSF_UNet_250705":                           SLK_MSF_UNet_250705(**BASE_ARGS),
        
        "DCLA_UNet_NoRes_250705":                        DCLA_UNet_NoRes_250705(**BASE_ARGS)
      
    },
    
    "DCLA_UNet 250708": {
        "SLK_UNet_250708":                               SLK_UNet_250708(**BASE_ARGS),
        
        "MSF_UNet_250708":                               MSF_UNet_250708(**BASE_ARGS),
        
        "SLK_MSF_UNet_250708":                           SLK_MSF_UNet_250708(**BASE_ARGS),
        
        "DCLA_UNet_NoRes_250708":                        DCLA_UNet_NoRes_250708(**BASE_ARGS)
      
    },
    
    "UltralightDCLAUNet": {
        "UltralightDCLAUNet":                               UltralightDCLAUNet(**BASE_ARGS),
    },
    "ConvNeXt": {
        # "ResUNeXt_Attn":                        ResUNeXt_Attn(**BASE_ARGS),
        
    }
    
    
    
}