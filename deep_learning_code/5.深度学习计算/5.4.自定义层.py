# -*- coding: utf-8 -*-
"""
@File    : 5.4.自定义层.py
@Time    : 2024/3/21 10:27
@Desc    : 
"""
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(2099)

"""
5.4.1 不带参数的自定义层
"""


class CenteredLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


"""
5.4.2 带参数的自定义层
"""


class MyLinear(nn.Module):
    def __init__(self, in_units, out_units):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(size=(in_units, out_units)))
        self.bias = nn.Parameter(torch.randn(out_units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight) + self.bias
        return F.relu(linear)


if __name__ == '__main__':
    X = torch.rand(size=(2, 5))

    # print("X = ", X)

    # layer = CenteredLayer()
    # print(layer(X))

    # net = nn.Sequential(nn.Linear(5, 10), CenteredLayer())
    # Y = net(X)
    # print(Y, Y.mean())

    # linear = MyLinear(5, 10)
    # print(linear)
    # print('weight: ', linear.weight, '\n\n', 'bias', linear.bias)
    # Y = linear(X)
    # print(Y, Y.shape)

    net = nn.Sequential(MyLinear(2, 5), MyLinear(5, 2))
    X1 = torch.rand(size=(3, 2))
    print(X1)
    print(net[0].state_dict())
    print(net(X1))
