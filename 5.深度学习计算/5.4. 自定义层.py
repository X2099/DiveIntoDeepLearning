# -*- coding: utf-8 -*-
"""
@File    : 5.4. 自定义层.py
@Time    : 2024/12/20 10:45
@Desc    : 
"""
import torch
from torch import nn
from torch.nn import functional as F


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()  # 将输入数据减去其均值


# layer = CenteredLayer()
# output = layer(torch.FloatTensor([1, 2, 3, 4, 5]))
# print(output)  # tensor([-2., -1.,  0.,  1.,  2.])

# net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())


# Y = net(torch.rand(4, 8))
# print(Y.mean())

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))  # 初始化权重
        self.bias = nn.Parameter(torch.randn(units))  # 初始化偏置

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data  # 计算线性变换
        return F.relu(linear)  # 使用ReLU激活函数


# linear = MyLinear(5, 3)
# print(linear.weight)
#
# output = linear(torch.randn(2, 5))
# print(output)


net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
output = net(torch.randn(2, 64))
print(output)
