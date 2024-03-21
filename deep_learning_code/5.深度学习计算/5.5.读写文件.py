# -*- coding: utf-8 -*-
"""
@File    : 5.5.读写文件.py
@Time    : 2024/3/21 11:07
@Desc    : 
"""
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(2099)

"""
5.5.1 保存和加载tensor
"""
# X = torch.randn(size=(3, 5))
# torch.save(X, 'X_file')  # tensor写入文件

# X1 = torch.load('X_file')
# print(X1)
# print(X1 == X)

a = torch.ones(5)
b = torch.zeros(5)
# torch.save([a, b], 'ab_file')

# print(torch.load('ab_file'))

# torch.save({'a': a, 'b': b}, 'ab_dict_file')

# print(torch.load('ab_dict_file'))

"""
5.5.2 保存和加载模型参数
"""


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


X = torch.rand((3, 20))
net = MLP()
Y = net(X)
# print(Y)

# 保存参数
# torch.save(net.state_dict(), 'mlp_net.params')

clone = MLP()
clone.load_state_dict(torch.load('mlp_net.params'))
# print(clone)

Y_clone = clone(X)

print(Y == Y_clone)
