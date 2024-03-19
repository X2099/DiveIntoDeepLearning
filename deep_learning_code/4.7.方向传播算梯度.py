# -*- coding: utf-8 -*-
"""
@File    : 4.7.方向传播算梯度.py
@Time    : 2024/3/19 13:55
@Desc    : 
"""
import torch
from pprint import pprint

torch.manual_seed(42)


def print_x_grad(x, label):
    print("*" * 100)
    print(f'{label} =', x)
    print(f'{label}.grad =', x.grad)
    print("*" * 100)


x = torch.arange(10, dtype=torch.float32).reshape(2, 5)
x.requires_grad = True

w1 = torch.normal(0, 1, size=(5, 1), requires_grad=True)

z = x @ w1

print_x_grad(x, 'x')
print_x_grad(w1, 'w1')

z.sum().backward()

print_x_grad(x, 'x')
print_x_grad(w1, 'w1')
