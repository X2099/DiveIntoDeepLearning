# -*- coding: utf-8 -*-
"""
@File    : 5.2.参数管理.py
@Time    : 2024/12/17 11:21
@Desc    : 
"""
from pprint import pprint

import torch
from torch import nn

shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8), nn.ReLU(),
    shared, nn.ReLU(),
    shared, nn.ReLU(),
    nn.Linear(8, 1)
)

# print(net[2].weight.data == net[4].weight.data)
"""
tensor([[True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True]])
"""
# print(net[2].bias.data == net[4].bias.data)
"""
tensor([True, True, True, True, True, True, True, True])
"""

net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0])
"""
tensor([ 1.0000e+02, -4.3048e-02, -2.2559e-01, -2.5617e-01,  2.4563e-01,
         7.6674e-02,  1.1903e-01,  2.4937e-01])
"""

print(net[2].weight.data[0] == net[4].weight.data[0])
"""
tensor([True, True, True, True, True, True, True, True])
"""


# net = nn.Sequential(
#     nn.Linear(4, 8),
#     nn.ReLU(),
#     nn.Linear(8, 1)
# )


def init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def my_init(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 42)


net[0].apply(init_xavier)
net[2].apply(init_42)

# net.apply(init_normal)

# print(net[0].weight.data)
# """
# tensor([[-0.4288, -0.0551,  0.5148, -0.6777],
#         [ 0.4790,  0.6359, -0.2876, -0.5548],
#         [ 0.4360, -0.0825, -0.4318, -0.2367],
#         [ 0.2139,  0.4846, -0.2292, -0.3446],
#         [ 0.1689, -0.1191,  0.4892,  0.5820],
#         [ 0.0284, -0.2763, -0.4984, -0.6381],
#         [ 0.4041, -0.6052,  0.3808, -0.1873],
#         [-0.1200, -0.5748,  0.1224, -0.1868]])
# """
# print(net[2].weight.data)
# """
# tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
# """
# print(net[0].bias.data)
"""
tensor([0., 0., 0., 0., 0., 0., 0., 0.])
"""


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net


# rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
#
# print(rgnet)
"""
Sequential(
  (0): Sequential(
    (block 0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)
"""

# print(rgnet[0][1][0].bias.data)
# tensor([-0.0140, -0.4541, -0.3257,  0.1100, -0.2222,  0.0812,  0.2502,  0.1304])

X = torch.rand(size=(2, 4))

# print(net(X))
"""
tensor([[-0.0241],
        [-0.0680]], grad_fn=<AddmmBackward0>)
"""

# pprint(net[2].state_dict())
"""
OrderedDict([('weight',
              tensor([[-0.2753,  0.2646,  0.1241,  0.3173,  0.1059,  0.1603,  0.0890,  0.0554]])),
             ('bias', tensor([-0.1769]))])
"""

# print(type(net[2].bias))  # <class 'torch.nn.parameter.Parameter'>
# print(net[2].bias)
"""
Parameter containing:
tensor([-0.1134], requires_grad=True)
"""
# print(net[2].bias.data)  # tensor([-0.1134])
#
# print('*' * 100)

# for name, parm in net.named_parameters():
#     print(name, parm.shape)
#     print(parm)
#     print('-' * 70)
"""
0.weight torch.Size([8, 4])
Parameter containing:
tensor([[-0.0557, -0.2581, -0.3506,  0.4918],
        [ 0.0967, -0.2477, -0.0037,  0.1130],
        [ 0.3352,  0.0911,  0.1741,  0.2439],
        [ 0.4347,  0.3009, -0.0760, -0.1117],
        [ 0.3735, -0.0265,  0.4541,  0.4448],
        [-0.2214,  0.0501, -0.2293,  0.2222],
        [ 0.3108, -0.1836,  0.0770,  0.4198],
        [ 0.0962, -0.2367,  0.0329,  0.4367]], requires_grad=True)
----------------------------------------------------------------------------------------------------
0.bias torch.Size([8])
Parameter containing:
tensor([ 0.2990, -0.1997, -0.4348, -0.1667,  0.2051, -0.4249, -0.1773,  0.0615],
       requires_grad=True)
----------------------------------------------------------------------------------------------------
2.weight torch.Size([1, 8])
Parameter containing:
tensor([[-0.1709,  0.1577, -0.0080, -0.0037, -0.1626, -0.0597,  0.2279,  0.3184]],
       requires_grad=True)
----------------------------------------------------------------------------------------------------
# 2.bias torch.Size([1])
Parameter containing:
tensor([0.0224], requires_grad=True)
----------------------------------------------------------------------------------------------------

"""
