# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/04/30 17:58:00
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 构建新的激活函数
*      VERSION: v1.0
*      FEATURES:  
               - Swish 激活函数
               
*      STATE:     
*      CHANGE ON: 
=================================================
'''


import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)