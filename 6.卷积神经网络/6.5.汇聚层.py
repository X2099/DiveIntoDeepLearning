# -*- coding: utf-8 -*-
"""
@File    : 6.5.汇聚层.py
@Time    : 2024/12/30 14:23
@Desc    : 
"""
import torch
from torch import nn

# 构造一个4x4的输入张量
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)

# 使用最大汇聚层，窗口大小为3x3，步幅为1
# pool2d = nn.MaxPool2d(kernel_size=3, stride=1)
# output = pool2d(X)
# print(output)

# 设置填充为1，步幅为2
# pool2d = nn.MaxPool2d(3, padding=1, stride=2)
# print(pool2d(X))

# 构造一个包含2个通道的输入张量
X = torch.cat((X, X + 1), dim=1)
print(X.shape)
print(X)

# 使用最大汇聚层处理2通道输入
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
