# -*- coding: utf-8 -*-
"""
@File    : 5.3.延后初始化.py
@Time    : 2024/3/20 15:30
@Desc    : 
"""
import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

print("输入维度未知时net[0]参数：", net[0].state_dict())  # 未初始化

X = torch.rand(size=(10, 64))

net(X)

print("输入确定后net[0]参数：", net[0].weight.data.shape)
