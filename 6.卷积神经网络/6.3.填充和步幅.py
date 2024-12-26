# -*- coding: utf-8 -*-
"""
@File    : 6.3.填充和步幅.py
@Time    : 2024/12/26 11:36
@Desc    : 
"""
import torch
from torch import nn

# 定义一个卷积层，卷积核大小为 3x3，填充为 1
# conv2d = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1)
# X = torch.rand(size=(8, 8))
# X = X.unsqueeze(0).unsqueeze(0)  # 添加批量维度和通道维度
# output = conv2d(X)
# print(output.shape)  # 输出为 torch.Size([1, 1, 8, 8])

# 定义卷积层，卷积核大小为 3x3，步幅为 2，填充为 1
# conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
# output = conv2d(torch.rand(size=(8, 8)).unsqueeze(0).unsqueeze(0))
# print(output.shape)  # torch.Size([1, 1, 4, 4])


# 定义卷积层，卷积核大小为 5x3，步幅为 (3, 2)，填充为 (2, 1)
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1), stride=(3, 2))
output = conv2d(torch.rand((8, 8)).unsqueeze(0).unsqueeze(0))
print(output.shape)  # 输出：torch.Size([1, 1, 3, 4])
