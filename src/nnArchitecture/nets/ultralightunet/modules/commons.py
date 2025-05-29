
import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
def norm_layer(name: str, num_channels: int, **kwargs):
    """
    获取指定类型的归一化层
    
    Args:
        name (str): 归一化类型 ('batch', 'instance', 'group', 'layer')
        num_channels (int): 输入通道数
        **kwargs: 其他参数（如 group_norm_groups）
    
    Returns:
        nn.Module: 对应的归一化层
    """
    if name == 'batch':
        return nn.BatchNorm3d(num_channels)
    elif name == "instance":
        affine = kwargs.get("instance_norm_affine", True)
        return nn.InstanceNorm3d(num_channels, affine=affine)
    elif name == "group":
        groups = kwargs.get("group_norm_groups", 4)  # 默认 4 组
        assert num_channels % groups == 0, "通道数必须能被组数整除"
        return nn.GroupNorm(groups, num_channels)
    elif name == "layer":
        return nn.LayerNorm(num_channels)
    else:
        raise ValueError(f"Unsupported normalization: {name}")
    
def act_layer(name: str, **kwargs):
    """
    获取指定类型的激活函数
    
    Args:
        name (str): 激活函数类型 ('relu', 'gelu', 'leaky_relu', 'swish', 'sigmoid')
        **kwargs: 其他参数（如 leaky_relu_slope)
    
    Returns:
        nn.Module: 对应的激活函数
    """
    if name == 'relu':
        return nn.ReLU()
    elif name == 'relu6':
        return nn.ReLU6()
    elif name == 'gelu':
        return nn.GELU()
    elif name == "leaky_relu":
        slope = kwargs.get("leaky_relu_slope", 0.1)
        return nn.LeakyReLU(negative_slope=slope)
    elif name == "swish":
        return Swish()
    elif name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {name}")
    
def channel_shuffle(x, groups):
    batchs, channels, depth, height, width = x.data.size()
    channels_per_group = channels // groups
    
    # reshape
    x = x.view(batchs, groups, channels_per_group, depth, height, width)
    
    x = torch.transpose(x, 1, 2).contiguous()
    
    # flatten
    x = x.view(batchs, -1, depth, height, width)
    
    return x    