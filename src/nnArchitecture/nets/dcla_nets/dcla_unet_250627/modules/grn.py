from .commons import *

class GRN3D(nn.Module):
    """
    Global Response Normalization for 3D ConvNets (ConvNeXt V2 style)
    适配输入形状: [B, C, D, H, W]
    """
    def __init__(self, channels, eps=1e-6):
        """
        Args:
            channels (int): 输入特征图的通道数
            eps (float): 防止除零的小常数
        """
        super().__init__()
        self.eps = eps
        # 通道缩放参数（gamma）和偏置（beta）
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        
        # 特征转换层（两层1x1x1卷积）
        self.fc1 = nn.Conv3d(channels, channels, 1)
        self.fc2 = nn.Conv3d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        """
        前向传播
        Args:
            x (Tensor): 输入特征图，形状 [B, C, D, H, W]
        Returns:
            Tensor: GRN处理后的特征图，形状不变
        """
        # 1. 全局特征聚合
        gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)  # L2范数聚合空间信息 [B, C, 1, 1, 1]
        
        # 2. 特征校准
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)   # 通道归一化
        
        # 3. 特征转换
        z = self.fc1(nx)
        z = self.act(z)
        z = self.fc2(z)
        
        # 4. 门控机制
        return x * (self.gamma * z.sigmoid() + self.beta + 1)    