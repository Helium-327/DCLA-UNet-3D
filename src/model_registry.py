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
    
    "DCLA_UNet v1": {
        "DCLA_UNet_v1":                         DCLA_UNet_v1(**BASE_ARGS),
        "ResUNetBaseline_S_DCLA_v1":            ResUNetBaseline_S_DCLA_v1(**BASE_ARGS),
        "ResUNetBaseline_S_SLK_v1":             ResUNetBaseline_S_SLK_v1(**BASE_ARGS),
       "ResUNetBaseline_S_DCLA_SLK_v1":        ResUNetBaseline_S_DCLA_SLK_v1(**BASE_ARGS),
        "ResUNetBaseline_S_MSF_v1":             ResUNetBaseline_S_MSF_v1(**BASE_ARGS),
        "ResUNetBaseline_S_DCLA_MSF_v1":        ResUNetBaseline_S_DCLA_MSF_v1(**BASE_ARGS),
    },
    
    
    "DCLA_UNet v2": {
        "DCLA_UNet_v2":                         DCLA_UNet_v2(**BASE_ARGS),
        "DCLA_UNet_v2_1":                       DCLA_UNet_v2_1(**BASE_ARGS),
        "DCLA_UNet_v2_2":                       DCLA_UNet_v2_2(**BASE_ARGS),
        "DCLA_UNet_v2_3":                       DCLA_UNet_v2_3(**BASE_ARGS),
        "ResUNetBaseline_S_DCLA_v2":            ResUNetBaseline_S_DCLA_v2(**BASE_ARGS),
        "ResUNetBaseline_S_DCLAv1_v2":          ResUNetBaseline_S_DCLAv1_v2(**BASE_ARGS),
        "ResUNetBaseline_S_SLK_v2":             ResUNetBaseline_S_SLK_v2(**BASE_ARGS),
        "ResUNetBaseline_S_SLKv1_v2":           ResUNetBaseline_S_SLKv1_v2(**BASE_ARGS),
        "ResUNetBaseline_S_SLKv2_v2":           ResUNetBaseline_S_SLKv2_v2(**BASE_ARGS),
        "ResUNetBaseline_S_DCLA_SLK_v2":        ResUNetBaseline_S_DCLA_SLK_v2(**BASE_ARGS),
        "ResUNetBaseline_S_DCLA_SLKv1_v2":      ResUNetBaseline_S_DCLA_SLKv1_v2(**BASE_ARGS),
        "ResUNetBaseline_S_DCLA_SLKv2_v2":      ResUNetBaseline_S_DCLA_SLKv2_v2(**BASE_ARGS),
        "ResUNetBaseline_S_DCLAv1_SLKv2_v2":    ResUNetBaseline_S_DCLAv1_SLKv2_v2(**BASE_ARGS),
        "ResUNetBaseline_S_MSF_v2":             ResUNetBaseline_S_MSF_v2(**BASE_ARGS),
        "ResUNetBaseline_S_DCLA_MSF_v2":        ResUNetBaseline_S_DCLA_MSF_v2(**BASE_ARGS),
        "ResUNetBaseline_S_DCLAv1_MSF_v2":      ResUNetBaseline_S_DCLAv1_MSF_v2(**BASE_ARGS),
        "ResUNetBaseline_S_SLK_MSF_v2":         ResUNetBaseline_S_SLK_MSF_v2(**BASE_ARGS),
        "ResUNetBaseline_S_SLKv1_MSF_v2":       ResUNetBaseline_S_SLKv1_MSF_v2(**BASE_ARGS),
        "ResUNetBaseline_S_SLKv2_MSF_v2":       ResUNetBaseline_S_SLKv2_MSF_v2(**BASE_ARGS),
        
        "DCLA_UNet_v2_4":                       DCLA_UNet_v2_4(**BASE_ARGS),
        "DCLA_UNet_v2_6":                       DCLA_UNet_v2_6(**BASE_ARGS),
        "DCLA_UNet_v2_7":                       DCLA_UNet_v2_7(**BASE_ARGS),
        "DCLA_UNet_withoutDCLA_v2_6":           DCLA_UNet_withoutDCLA_v2_6(**BASE_ARGS),    
        "DCLA_UNet_withoutDCLA_v2_7":           DCLA_UNet_withoutDCLA_v2_7(**BASE_ARGS), 
    },
    
    "DCLA_UNet v3":{
        "DCLA_UNet_v3":                         DCLA_UNet_v3(**BASE_ARGS),
        "ResUNetBaseline_S_DCLA_SLKv2_v3":      ResUNetBaseline_S_DCLA_SLKv2_v3(**BASE_ARGS),
        "ResUNetBaseline_S_MSF_v3":             ResUNetBaseline_S_MSF_v3(**BASE_ARGS),
        "ResUNetBaseline_S_DCLA_MSF_v3":        ResUNetBaseline_S_DCLA_MSF_v3(**BASE_ARGS),
        "ResUNetBaseline_S_SLKv2_MSF_v3":       ResUNetBaseline_S_SLKv2_MSF_v3(**BASE_ARGS),
        "ResUNetBaseline_S_SLKv2_v3":           ResUNetBaseline_S_SLKv2_v3(**BASE_ARGS),    
        
    },
    
    "DCLA_UNet v4": {

        "DCLA_UNet_finalv4":                    DCLA_UNet_finalv4(**BASE_ARGS)
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
    
    "DCLA_UNet 250602": {
        "DCLA_UNet_250602":                     DCLA_UNet_250602(**BASE_ARGS),
        "BaseLine_S_DCLA_250602":               BaseLine_S_DCLA_250602(**BASE_ARGS),
    },

    "DCLA_UNet 250603": {
        "DCLA_UNet_250603":                     DCLA_UNet_250603(**BASE_ARGS),
        "DCLA_UNet_250603_v2":                  DCLA_UNet_250603_v2(**BASE_ARGS),
        "DCLA_UNet_250603_v3":                  DCLA_UNet_250603_v3(**BASE_ARGS),
        "DCLA_UNet_250603_v4":                  DCLA_UNet_250603_v4(**BASE_ARGS),
        "BaseLine_S_DCLA_250603":               BaseLine_S_DCLA_250603(**BASE_ARGS),
    },
    
    "DCLA_UNet 250604": {
      "ResUNeXt":                               ResUNeXt(**BASE_ARGS),
      "DCLA_UNet_250604":                       DCLA_UNet_250604(**BASE_ARGS),
      "DCLA_UNet_withoutDCLA_250604":           DCLA_UNet_withoutDCLA_250604(**BASE_ARGS),
      "Base_ResNeXt_250604":                    Base_ResNeXt_250604(**BASE_ARGS),
      "Base_ResNeXt_DCLA_250604":               Base_ResNeXt_DCLA_250604(**BASE_ARGS),
      "Base_MSF_250604":                        Base_MSF_250604(**BASE_ARGS),
      "Base_MSF_DCLA_250604":                   Base_MSF_DCLA_250604(**BASE_ARGS),
    },
    "DCLA_UNet 250605": {
    #   "ResUNeXt":                               ResUNeXt(**BASE_ARGS),
      "DCLA_UNet_250605":                         DCLA_UNet_250605(**BASE_ARGS),
    #   "DCLA_UNet_250605v2":                       DCLA_UNet_250605v2(**BASE_ARGS),
      "DCLA_UNet_withoutDCLA_250605":           DCLA_UNet_withoutDCLA_250605(**BASE_ARGS),
      "Base_ResNeXt_250605":                    Base_ResNeXt_250605(**BASE_ARGS),
      "Base_ResNeXt_DCLA_250605":               Base_ResNeXt_DCLA_250605(**BASE_ARGS),
      "Base_MSF_250605":                        Base_MSF_250605(**BASE_ARGS),
      "Base_MSF_DCLA_250605":                   Base_MSF_DCLA_250605(**BASE_ARGS),
    },
    "DCLA_UNet 250606": {
    #   "ResUNeXt":                               ResUNeXt(**BASE_ARGS),
      "DCLA_UNet_250606":                         DCLA_UNet_250606(**BASE_ARGS),
    #   "DCLA_UNet_250605v2":                       DCLA_UNet_250605v2(**BASE_ARGS),
      "DCLA_UNet_withoutDCLA_250606":           DCLA_UNet_withoutDCLA_250606(**BASE_ARGS),
      "Base_ResNeXt_250606":                    Base_ResNeXt_250606(**BASE_ARGS),
      "Base_ResNeXt_DCLA_250606":               Base_ResNeXt_DCLA_250606(**BASE_ARGS),
      "Base_MSF_250606":                        Base_MSF_250606(**BASE_ARGS),
      "Base_MSF_DCLA_250606":                   Base_MSF_DCLA_250606(**BASE_ARGS),
    },
    
    "ConvNeXt": {
        # "ResUNeXt_Attn":                        ResUNeXt_Attn(**BASE_ARGS),
        
    }
    
    
    
}