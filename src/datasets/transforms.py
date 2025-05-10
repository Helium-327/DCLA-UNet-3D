# -*- coding: UTF-8 -*-
'''

Describle:         数据增强探索

Created on         2024/07/31 16:10:16
Author:            @ Mr_Robot
Current State:     # TODO: 
                    1. 添加 monai 的数据增强
                    2. 添加 torchio 的数据增强
                    3. 添加 albumentation 的数据增强
'''

import torch
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import rotate

seed = 42#seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)#numpy产生的随机数一致


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, vimage, vmask):
        for t in self.transforms:
            vimage, vmask = t(vimage, vmask)
        return vimage, vmask
    def returnName(self):
        return [t.__class__.__name__ for t in self.transforms]
    
class ToTensor(object):
    def __init__(self):
        pass
    
    def __call__(self, vimage, vmask):
        if isinstance(vimage, np.ndarray):
            vimage = torch.tensor(vimage.copy()).float()
        if isinstance(vmask.copy(), np.ndarray):
            vmask = torch.tensor(vmask.copy()).long()  #! .copy()去掉会出现报错：负步长
        return vimage, vmask
    
""" 随机裁剪 """
class RandomCrop3D(object):
    def __init__(self, size=(128, 128, 128)):
        self.size = size
        
    def __call__(self, vimage, vmask):
        img = np.array(vmask)
        x_start = np.random.randint(0, img.shape[-3] - self.size[0]) if img.shape[-3] > self.size[0] else 0
        y_start = np.random.randint(0, img.shape[-2] - self.size[1]) if img.shape[-2] > self.size[1] else 0
        z_start = np.random.randint(0, img.shape[-1] - self.size[2]) if img.shape[-1] > self.size[2] else 0
        
        vimage = vimage[:,x_start: x_start + self.size[0], y_start: y_start + self.size[1], z_start: z_start + self.size[2]]
        vmask = vmask[x_start: x_start + self.size[0], y_start: y_start + self.size[1], z_start: z_start + self.size[2]]
        return vimage, vmask

""" 中心裁剪 """
class CenterCrop3D:
    def __init__(self, output_size):
        self.output_size = np.array(output_size)

    def __call__(self, image, mask):
        input_shape = np.array(image.shape[1:])  # Assuming channel-first format
        start = (input_shape - self.output_size) // 2
        end = start + self.output_size
        slices = tuple(slice(s, e) for s, e in zip(start, end))
        return image[(slice(None),) + slices], mask[slices]
    
""" 归一化处理 """
class FrontGroundNormalize(object):
    def __init__(self):
        pass
    
    def __call__(self, vimage, vmask):
        mask = np.sum(vimage, axis=0) > 0 
        for k in range(4):
            x = vimage[k, ...]
            y = x[mask]
            if y.size > 0:
                x[mask] = (x[mask] - np.mean(y)) / (np.std(y) + 1e-6)
            vimage[k, ...] = x
        return vimage, vmask
    
class RandomFlip3D(object):
    """3D随机翻转增强"""
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): 沿任意轴翻转的概率
        """
        self.prob = prob

    def __call__(self, image, mask):
        # 对每个空间维度独立判断是否翻转
        for dim in range(3):
            if np.random.rand() < self.prob:
                image = np.flip(image, axis=dim+1)  # 图像维度为(C,H,W,D)
                mask = np.flip(mask, axis=dim)
        return image, mask

class RandomNoise3D(object):
    def __init__(self, mean=0.0, std=(0, 0.25)):
        self.mean = mean
        self.std = std

    def __call__(self, vimage, vmask):
        # 生成标准差σ：如果std是区间则随机采样
        if isinstance(self.std, (tuple, list)):
            sigma = np.random.uniform(self.std[0], self.std[1])
        else:
            sigma = self.std
        # 生成高斯噪声：μ=mean，σ=随机值，形状与输入数组相同
        noise = self.mean + sigma * np.random.randn(*vimage.shape)
        # 叠加噪声并返回
        return vimage + noise, vmask

# class CustomRandomFlip3D(object):
#     def __init__(self, p=0.5, axes=("LR", "AP")):
#         self.p = p
#         self.axes_mapping = {"LR": -1, "AP": -2}
#         self.axes_indices = []
        
#         for axis in axes:
#             if isinstance(axis, str):
#                 if axis not in self.axes_mapping:
#                     raise ValueError(f"Invalid axis name: {axis}. Use 'LR', 'AP', 'IS' or integers.")
#                 self.axes_indices.append(self.axes_mapping[axis])
#             elif isinstance(axis, int):
#                 self.axes_indices.append(axis)
#             else:
#                 raise ValueError(f"Invalid axis type: {type(axis)}. Use string or integer.")

#     def __call__(self, image, mask):
#         flip_image = image.clone()
#         flip_mask = mask.clone()
        
#         # 确保至少翻转一次
#         for axis in self.axes_indices:
#             if torch.rand(1).item() < self.p:
#                 flip_image = torch.flip(flip_image, dims=[axis])
#                 flip_mask = torch.flip(flip_mask, dims=[axis if axis < 0 else axis-1])
        
#         return flip_image, flip_mask

class RandomRotation3D(object):
    def __init__(self, degrees=(-10, 10), axes=(0, 1, 2), p=0.5):
        """
        初始化 CustomRandomRotation3D 类
        
        参数:
        - degrees: 旋转角度范围，可以是一个元组 (min_angle, max_angle) 或者一个浮点数
        - axes: 要进行旋转的轴，可以是 (0, 1, 2) 中的任意组合
        - p: 每个轴应用旋转的概率
        """
        self.degrees = degrees if isinstance(degrees, tuple) else (-degrees, degrees)
        self.axes = axes
        self.p = p

    def __call__(self, image, mask):
        """
        对输入的图像和掩码进行随机旋转
        
        参数:
        - image: 形状为 (C, D, H, W) 的 4D numpy array
        - mask: 形状为 (D, H, W) 的 3D numpy array
        
        返回:
        - 旋转后的图像和掩码
        """
        rotated_image = image.copy()
        rotated_mask = mask.copy()

        for axis in self.axes:
            if np.random.rand() < self.p:
                angle = np.random.uniform(self.degrees[0], self.degrees[1])
                rotated_image = self.rotate(rotated_image, angle, axis)
                rotated_mask = self.rotate(rotated_mask, angle, axis, is_mask=True)

        return rotated_image, rotated_mask

    def rotate(self, x, angle, axis, is_mask=False):
        """
        沿指定轴旋转数组
        """
        axes = [(1, 2), (0, 2), (0, 1)][axis]
        if is_mask:
            return rotate(x, angle, axes=axes, reshape=False, order=0, mode='nearest')
        else:
            return np.stack([rotate(channel, angle, axes=axes, reshape=False, order=1, mode='constant', cval=0) 
                             for channel in x])
    
class ScaleIntensityRanged(object):
    """MRI强度值归一化到制定范围(通道独立处理)"""
    def __init__(self, a_min=-100.0, 
                 a_max=3000.0, b_min=0.0, b_max=1.0
                 ):
        """
        Args:
            a_min(float):原数据截断下限
            a_max(float):原始数据截断上限
            b_min(float):归一化下限
            b_max(float):归一化上限
        """
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        
    def __call__(self, image, mask):
        """
        Args:
            vimage(np.ndarray):输入图像
            vmask(np.ndarray):输入mask
        """
        
        for c in range(image.shape[0]):
            channel = image[c]
            # 1.截断异常值
            channel = np.clip(channel, self.a_min, self.a_max)
            # 2. 线性映射到目标范围
            if self.a_max != self.a_min:
                channel = (channel - self.a_min) /(self.a_max - self.a_min)
                channel = channel * (self.b_max - self.b_min) + self.b_min
            image[c] = channel
            
        return image, mask
    
if __name__ == "__main__":
    pass
    # CropSize = (128, 128, 128)
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # trans = Compose([
    #     RandomCrop3D(size=CropSize),
    #     FrontGroundNormalize(mean=mean, std=std),
    #     tioRandomFlip3d(),
    #     tioRandomElasticDeformation3d(),
    #     # tioZNormalization(),
    #     # tioRandomNoise3d(),
    #     tioRandomGamma3d()
    # ])
    
    # data = torch.randn(4, 240, 240, 155)
    # mask = torch.randint(0, 4, (240, 240, 155))
    # data, mask = trans(data, mask)
    # print(data.shape, mask.shape)
    
    # print(trans.returnName())

    
    
