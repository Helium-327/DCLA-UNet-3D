from .baselines import (
    RA_UNet,
    DWResUNet,
    ResUNetBaseline_S,
    ResUNetBaseline_M
)

from .sota_nets import (
    UNet3D,
    AttUNet3D,
    UNETR,
    UNETR_PP,
    SegFormer3D,
    # Mamba3d,
    # MogaNet
)
from .dcla_nets import (
    DCLA_UNet_v1,
    ResUNetBaseline_S_DCLA_v1,
    ResUNetBaseline_S_SLK_v1,
    ResUNetBaseline_S_DCLA_SLK_v1,
    ResUNetBaseline_S_MSF_v1,
    ResUNetBaseline_S_DCLA_MSF_v1,
    ResUNetBaseline_S_SLK_MSF_v1,
    
    DCLA_UNet_v2,
    DCLA_UNet_v2_1,
    DCLA_UNet_v2_2,
    DCLA_UNet_v2_3,
    
    ResUNetBaseline_S_DCLA_v2,
    ResUNetBaseline_S_DCLAv1_v2,
    ResUNetBaseline_S_SLK_v2,
    ResUNetBaseline_S_SLKv1_v2,
    ResUNetBaseline_S_SLKv2_v2,
    ResUNetBaseline_S_DCLA_SLK_v2,
    ResUNetBaseline_S_DCLA_SLKv1_v2,
    ResUNetBaseline_S_DCLA_SLKv2_v2,
    ResUNetBaseline_S_DCLAv1_SLKv2_v2,
    ResUNetBaseline_S_MSF_v2,
    ResUNetBaseline_S_DCLA_MSF_v2,
    ResUNetBaseline_S_DCLAv1_MSF_v2,
    ResUNetBaseline_S_SLK_MSF_v2,
    ResUNetBaseline_S_SLKv1_MSF_v2,
    ResUNetBaseline_S_SLKv2_MSF_v2
)


from .slim_sep_unets import (
    # SlimSepUNetv1
    SlimSepUNetv1,
    SlimSepUNetv1_LKDCLA,
    
    # SlimSepUNetv2
    SlimSepUNetv2,
    SlimSepUNetv2_LKDCLA,
    
    # SlimSepUNetv3
    SlimSepUNetv3,
    SlimSepUNetv3_LKDCLA,
    
    # SlimSepUNetv4
    SlimSepUNetv4,
    SlimSepUNetv4_LiteDASPP,
    SlimSepUNetv4_LiteDASPP_LKDCLA,
    
    # SlimSepUNetv5
    SlimSepUNetv5,
    SlimSepUNetv5_LiteDASPP,
    SlimSepUNetv5_LiteDASPP_LKDCLA
)

from .mega_res_unets import (
    MegaResUNetv1,
    MegaResUNetv1_DCLA,
    MegaResUNetv1_DCLA_AG,
    MegaResUNetv1_DCLA_MADG,
    
    MegaResUNetv2,
    MegaResUNetv2_DCLA,
    MegaResUNetv2_DCLA_AG,
    MegaResUNetv2_DCLA_MADG,
    
    MegaResUNetv3,
    MegaResUNetv3_DCLA,
    MegaResUNetv3_LKDCLA,
    MegaResUNetv3_SLKDCLA,
    
    MegaResUNetv4,
    MegaResUNetv4_DCLA,
    MegaResUNetv4_LKDCLA,
    MegaResUNetv4_SLKDCLA,
    
    MegaResUNetv5,
    MegaResUNetv5_DCLA,
    MegaResUNetv5_LKDCLA,
    MegaResUNetv5_SLKDCLA,
    
    MegaResUNetv6,
    MegaResUNetv6_DCLA,
    MegaResUNetv6_LKDCLA,
    MegaResUNetv6_SLKDCLA,
    
    MegaResUNetv7,
    MegaResUNetv7_LKDCLA,
    
    MegaResUNetv8,
    MegaResUNetv8_LKDCLA,
    
    MegaResUNetv9,
    MegaResUNetv9_LKDCLA
)

from .res_unets import (
    ResUNet3D_S,
    ResUNet_S_DCLA,
    ResUNet_S_DASPP,
    ResUNet_S_MADG_DASPP,
    ResUNet_S_DCLA_AG,
    ResUNet_S_DCLA_AG_DASPP,
    ResUNet_S_MADG_DASPP_DCLA,
    ResUNet_S_DCLA_MADG,
    ResUNet_S_DCLA_MADG_DASPP,
    ResUNet3D_M,
    ResUNet_M_DASPP,
    ResUNet_M_DCLA,
    ResUNet_M_DCLA_AG,
    ResUNet_M_DCLA_AG_DASPP,
    ResUNet_M_DCLA_MADG,
    ResUNet_M_DCLA_MADG_DASPP,
    ResUNet_M_MADG_DASPP,
    ResUNet_M_MADG_DASPP_DCLA
)