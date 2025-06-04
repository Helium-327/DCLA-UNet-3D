
import torch
import torch.nn as nn

from mamba_ssm import Mamba
from torch.amp import autocast

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    @autocast(enabled=False, device_type="cuda")
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out


class AxialAttention3D(nn.Module):
    def __init__(self, in_dim, q_k_dim, patch_ini,axis='D'):
        """
        Parameters
        ----------
        in_dim : int
            channel of input tensor
        q_k_dim : int
            channel of Q, K vector
        axis : str
            attention axis, can be 'D', 'H', or 'W'
        """
        super(AxialAttention3D, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.axis = axis
        D,H,W=patch_ini[0],patch_ini[1],patch_ini[2]


        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        if self.axis == 'D':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, D, 1, 1))
        elif self.axis == 'H':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, H, 1))
        elif self.axis == 'W':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, 1, W))
        else:
            raise ValueError("Axis must be one of 'D', 'H', or 'W'.")
        
        nn.init.xavier_uniform_(self.pos_embed)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x,processed):
        """
        Parameterse
        ----------
        x : Tensor
            5-D tensor, (batch, channels, depth, height, width)
        """
        B, C, D, H, W = x.size()
        
        Q = self.query_conv(processed) + self.pos_embed  # (B, q_k_dim, D, H, W) + pos_embed
        K = self.key_conv(processed) + self.pos_embed  # (B, q_k_dim, D, H, W) + pos_embed
        V = self.value_conv(processed)  # (B, in_dim, D, H, W)
        # Q = self.query_conv(x)  # (B, q_k_dim, D, H, W)
        # K = self.key_conv(x)   # (B, q_k_dim, D, H, W)
        # V = self.value_conv(x)  # (B, in_dim, D, H, W)
        scale = math.sqrt(self.q_k_dim)
        if self.axis == 'D':
            Q = Q.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, D, q_k_dim)
            Q = Q.view(B*H*W, D, self.q_k_dim)  # (B*H*W, D, q_k_dim)
            
            K = K.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, q_k_dim, D)
            K = K.view(B*H*W, self.q_k_dim, D)  # (B*H*W, q_k_dim, D)
            
            V = V.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, D, in_dim)
            V = V.view(B*H*W, D, self.in_dim)  # (B*H*W, D, in_dim)
            
            attn = torch.bmm(Q, K) / scale # (B*H*W, D, D)
            attn = self.softmax(attn)
            
            out = torch.bmm(attn, V)  # (B*H*W, D, in_dim)

            out = out.view(B, H, W, D, self.in_dim)
            out = out.permute(0, 4, 3, 1, 2).contiguous()  # (B, C, D, H, W)
            
        elif self.axis == 'H':
            Q = Q.permute(0, 2, 4, 3, 1).contiguous()  # (B, D, W, H, q_k_dim)
            Q = Q.view(B*D*W, H, self.q_k_dim)  # (B*D*W, H, q_k_dim)
            
            K = K.permute(0, 2, 4, 1, 3).contiguous()  # (B, D, W, q_k_dim, H)
            K = K.view(B*D*W, self.q_k_dim, H)  # (B*D*W, q_k_dim, H)
            
            V = V.permute(0, 2, 4, 3, 1).contiguous()  # (B, D, W, H, in_dim)
            V = V.view(B*D*W, H, self.in_dim)  # (B*D*W, H, in_dim)
            
            attn = torch.bmm(Q, K) / scale # (B*D*W, H, H)
            attn = self.softmax(attn)
            
            out = torch.bmm(attn, V)  # (B*D*W, H, in_dim)
            out = out.view(B, D, W, H, self.in_dim)
            out = out.permute(0, 4, 1, 3, 2).contiguous()  # (B, C, D, H, W)
            
        else:  # self.axis == 'W'
            Q = Q.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, q_k_dim)
            Q = Q.view(B*D*H, W, self.q_k_dim)  # (B*D*H, W, q_k_dim)
            
            K = K.permute(0, 2, 3, 1, 4).contiguous()  # (B, D, H, q_k_dim, W)
            K = K.view(B*D*H, self.q_k_dim, W)  # (B*D*H, q_k_dim, W)
            
            V = V.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, in_dim)
            V = V.view(B*D*H, W, self.in_dim)  # (B*D*H, W, in_dim)
            
            attn = torch.bmm(Q, K) / scale # (B*D*H, W, W)
            attn = self.softmax(attn)
            
            out = torch.bmm(attn, V)  # (B*D*H, W, in_dim)
            out = out.view(B, D, H, W, self.in_dim)
            out = out.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)
        
        gamma = torch.sigmoid(self.gamma)
        out = gamma * out + (1-gamma) * x
        return out


class DirectionalMamba(nn.Module):
    def __init__(self, d_model,patch_ini ,d_state=32, axis='D', is_vssb=True, is_slice_attention=True):
        super().__init__()
        self.is_vssb = is_vssb
        self.axis = axis
        
        self.permute_dict = {
            'D': [(0, 1, 2, 3, 4), (0, 2, 3, 4, 1), (0, 4, 1, 2, 3)],  # 原始->mamba输入->输出
            'H': [(0, 1, 2, 3, 4), (0, 3, 2, 4, 1), (0, 4, 2, 1, 3)],
            'W': [(0, 1, 2, 3, 4), (0, 4, 2, 3, 1), (0, 4, 2, 3, 1)]
        }
        
        if self.is_vssb:
            self.mamba = VSSBlock(hidden_dim=d_model, d_state=d_state)
        else:
            self.mamba = MambaLayer(dim=d_model, d_state=d_state)

        if is_slice_attention:
            # self.slice_attention = MultiHeadAxialAttention3D(in_dim=d_model, q_k_dim=d_model, axis=axis,patch_ini=patch_ini)
            self.slice_attention = AxialAttention3D(in_dim=d_model, q_k_dim=d_model, axis=axis,patch_ini=patch_ini)
        else:
            self.slice_attention = nn.Identity()
        
    def forward(self, x):
        # x: [B, C, D, H, W]
        
        original_order, to_mamba, from_mamba = self.permute_dict[self.axis]
        
        if self.is_vssb:

            x_mamba = x.permute(*to_mamba)  # [B, L, *, *, C]，L是处理维度(D/H/W)
            B, L = x_mamba.shape[:2]
            x_mamba = x_mamba.reshape(B * L, *x_mamba.shape[2:])
            
            processed = self.mamba(x_mamba)
            
            processed = processed.reshape(B, L, *processed.shape[1:])
            processed = processed.permute(*from_mamba)  # [B, C, D, H, W]
        else :
            x_mamba = x.permute(*to_mamba)  # [B, L, *, *, C]，L是处理维度(D/H/W)
            B, L = x_mamba.shape[:2]
            x_mamba = x_mamba.reshape(B * L, x_mamba.shape[-1],*x_mamba.shape[2:-1])
            processed = self.mamba(x_mamba)
             

        if isinstance(self.slice_attention, nn.Identity):
            return processed
        else:
            attn_result = self.slice_attention(x,processed)
            return attn_result


class TriplaneMamba3DConcat(nn.Module):
    def __init__(self, input_channels,patch_ini,reduce_factor=1,is_slice_attention=True,is_shuffle=False,is_split=True,is_res=True,is_proj=False):
        super().__init__()
        
        hidden_channels = input_channels
        self.is_proj=is_proj
        if self.is_proj:
            self.proj_in = nn.Sequential(
            nn.Conv3d(hidden_channels, input_channels, 1,stride=1, padding=0),
            nn.LeakyReLU(),
            nn.InstanceNorm3d(input_channels)
        )
            hidden_channels = input_channels//reduce_factor
        self.hidden_channels = hidden_channels
        self.is_res=is_res
        self.is_shuffle=is_shuffle
        self.is_split=is_split
        if self.is_split:
            self.mamba_x = DirectionalMamba(d_model=hidden_channels//2,axis="D",patch_ini=patch_ini,is_slice_attention=is_slice_attention)
            self.mamba_y = DirectionalMamba(d_model=hidden_channels//4,axis="H",patch_ini=patch_ini,is_slice_attention=is_slice_attention)
            self.mamba_z = DirectionalMamba(d_model=hidden_channels//4,axis="W",patch_ini=patch_ini,is_slice_attention=is_slice_attention)
        else:
            self.mamba_x = DirectionalMamba(d_model=hidden_channels,axis="D",patch_ini=patch_ini,is_slice_attention=is_slice_attention)
            self.mamba_y = DirectionalMamba(d_model=hidden_channels,axis="H",patch_ini=patch_ini,is_slice_attention=is_slice_attention)
            self.mamba_z = DirectionalMamba(d_model=hidden_channels,axis="W",patch_ini=patch_ini,is_slice_attention=is_slice_attention)

        self.fusion = nn.Sequential(
            nn.Conv3d(hidden_channels, input_channels, 1,stride=1, padding=0),
            nn.LeakyReLU(),
            nn.InstanceNorm3d(input_channels)
        )
    def forward(self, x):
        # x shape: [B, C, D, H, W]
        if self.is_proj:
            x = self.proj_in(x)
        C=self.hidden_channels
        if self.is_split:
            channel_quarter = C // 4
            x_feat = x[:, :channel_quarter*2]  # 1/2
            y_feat = x[:, channel_quarter*2:channel_quarter*3]  # 1/4
            z_feat = x[:, channel_quarter*3:]  # 1/4

            feat_list = []
            
            feat_list.append(self.mamba_x(x_feat))

            feat_list.append(self.mamba_y(y_feat))

            feat_list.append(self.mamba_z(z_feat))
            
            # feat_list.append(res_feat)
            out = self.fusion(torch.cat(feat_list, dim=1))
            if self.is_shuffle:
                ShuffleChannelBlock = ShuffleChannel(groups=4)
                out = ShuffleChannelBlock(out)
        else:
            feat_list = []
            
            feat_list.append(self.mamba_x(x))

            feat_list.append(self.mamba_y(x))

            feat_list.append(self.mamba_z(x))
            
            out = self.fusion(feat_list[0]+feat_list[1]+feat_list[2])
        if self.is_res:
            return out + x
        else:
            return out