# -*- coding: utf-8 -*-
"""
@File    : debug.py
@Time    : 2024/3/8 10:18
@Desc    : 
"""
import torch
from torch import nn

torch.manual_seed(2099)

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

X = torch.rand()

nn.init.normal_