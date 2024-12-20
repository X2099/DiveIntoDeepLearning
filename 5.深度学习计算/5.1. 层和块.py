# -*- coding: utf-8 -*-
"""
@File    : 5.1. 层和块.py
@Time    : 2024/12/16 15:47
@Desc    : 
"""
from pprint import pprint

import torch
from torch import nn
from torch.nn import functional as F


class FixedHiddenMLP(nn.Module):
    """自定义有固定参数的块"""

    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)  # 不计算梯度的常数权重
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            print("X.abs().sum() =", X.abs().sum())
            X /= 2
        print("X.abs().sum() =", X.abs().sum())
        return X.sum()


# 测试固定权重的网络
net = FixedHiddenMLP()
X = torch.rand(2, 20)
print(net(X))
"""
X.abs().sum() = tensor(12.0223, grad_fn=<SumBackward0>)
X.abs().sum() = tensor(6.0112, grad_fn=<SumBackward0>)
X.abs().sum() = tensor(3.0056, grad_fn=<SumBackward0>)
X.abs().sum() = tensor(1.5028, grad_fn=<SumBackward0>)
X.abs().sum() = tensor(0.7514, grad_fn=<SumBackward0>)
tensor(0.2706, grad_fn=<SumBackward0>)
"""


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


# 使用自定义的Sequential实现
# net = MySequential(
#     nn.Linear(20, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10)
# )
# pprint(net._modules)
# """
# OrderedDict([('0', Linear(in_features=20, out_features=256, bias=True)),
#              ('1', ReLU()),
#              ('2', Linear(in_features=256, out_features=10, bias=True))])
# """
# X = torch.rand(2, 20)
#
# print(net(X))
"""
tensor([[ 0.1490,  0.1756, -0.0927, -0.1953,  0.1193,  0.1560,  0.0340, -0.1926,
          0.2319, -0.4672],
        [-0.0448,  0.0712, -0.0696, -0.1076,  0.0780,  0.0239, -0.1082, -0.2056,
          0.1978, -0.3106]], grad_fn=<AddmmBackward0>)
"""
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(20, 256)  # 隐藏层
#         self.out = nn.Linear(256, 10)  # 输出层
#
#     def forward(self, X):
#         """前向传播计算"""
#         return self.out(F.relu(self.hidden(X)))


# 实例化并测试网络
# net = MLP()
# X = torch.rand(2, 20)
#
# print(X.shape)  # torch.Size([2, 20])
#
# print(net(X))
"""
tensor([[ 0.0155, -0.1311, -0.0113, -0.3112,  0.0482,  0.1557, -0.2504,  0.0271,
         -0.1276, -0.0866],
        [ 0.0543, -0.1713, -0.0039, -0.2339,  0.0922,  0.1267, -0.2425,  0.0483,
         -0.0868, -0.1099]], grad_fn=<AddmmBackward0>)
"""

# net = nn.Sequential(
#     nn.Linear(20, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10)
# )
#
# X = torch.rand(2, 20)
# print(X)
# """
# tensor([[0.3957, 0.5312, 0.8056, 0.3749, 0.3407, 0.2381, 0.9757, 0.5841, 0.6490,
#          0.1689, 0.2499, 0.2440, 0.0397, 0.7120, 0.8425, 0.2046, 0.1753, 0.4144,
#          0.7518, 0.6320],
#         [0.9160, 0.1758, 0.8196, 0.2785, 0.4392, 0.7909, 0.8659, 0.0298, 0.1440,
#          0.9179, 0.3337, 0.4030, 0.5683, 0.0869, 0.7739, 0.7054, 0.8481, 0.7669,
#          0.7208, 0.5828]])
# """
# print(net(X))
"""
tensor([[-0.0696, -0.0993, -0.0359, -0.3313,  0.1255,  0.1074, -0.1138,  0.1102,
         -0.0300, -0.0882],
        [ 0.0118, -0.0319, -0.0921, -0.3144,  0.0938,  0.0856, -0.1590,  0.1882,
         -0.0626, -0.1599]], grad_fn=<AddmmBackward0>)
"""
