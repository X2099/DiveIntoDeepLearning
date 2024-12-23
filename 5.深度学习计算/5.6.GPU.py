# -*- coding: utf-8 -*-
"""
@File    : 5.6.GPU.py
@Time    : 2024/12/23 14:41
@Desc    : 
"""
import torch
from torch import nn

net = nn.Sequential(nn.Linear(3, 1))
X = torch.randn((2, 3))
output = net(X)
print(output.device)

X = X.cuda()
net = net.to(device=torch.device('cuda'))
output = net(X)
print(output.device)

X1 = torch.ones(2, 3)

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:0')

torch.cuda.device_count()


def try_gpu(i=0):
    """
    如果存在，则返回gpu(i)，否则返回cpu()
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """
    返回所有可用的GPU，如果没有GPU，则返回[cpu(),]
    """
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())
               ]
    return devices if devices else [torch.device('cpu'), ]

# print(try_gpu(), try_gpu(1), try_all_gpus())
#
# X = torch.ones(2, 3, device='cuda:0')
#
# x = torch.tensor([1, 2, 3])
