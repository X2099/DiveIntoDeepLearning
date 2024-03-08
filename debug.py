# -*- coding: utf-8 -*-
"""
@File    : debug.py
@Time    : 2024/3/8 10:18
@Desc    : 
"""
import torch

# help(torch.normal)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)

# help(W.sum)

num_classes = [1]
one_hot_matrix = torch.eye(3)
print(one_hot_matrix[1])
