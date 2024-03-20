# -*- coding: utf-8 -*-
"""
@File    : 5.1.层和块.py
@Time    : 2024/3/20 10:00
@Desc    : 
"""
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(2099)

X = torch.rand(2, 20)

print('-' * 100)
print("X = ", X, X.shape)
print('-' * 100)


# 5.1.1 自定义块
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


# 5.1.2 自定义顺序块
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


# 5.1.3 在前向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机参数
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.matmul(X, self.rand_weight) + 1)
        X = self.linear(X)  # 复用线性层，共享参数
        # 自定义控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


# 5.1.4 混合搭配各种组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


if __name__ == '__main__':
    # net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    # net = MLP()
    # net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    # net = FixedHiddenMLP()
    net = NestMLP()
    print(net)
    Y = net(X)
    print(Y, Y.shape)
