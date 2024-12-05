# -*- coding: utf-8 -*-
"""
@File    : 4.1.多层感知机.py
@Time    : 2024/11/14 17:16
@Desc    : 
"""
import torch

# 示例输入
x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
# 应用 ReLU 激活函数
relu_output = torch.relu(x)
print(relu_output)  # 输出：tensor([0., 0., 1., 2.])

# 示例输入
x = torch.tensor([0.0, 1.0, -1.0])
# 应用 Sigmoid 激活函数
sigmoid_output = torch.sigmoid(x)
print(sigmoid_output)  # 输出：tensor([0.5000, 0.7311, 0.2689])

# 示例输入
x = torch.tensor([-1.0, 0.0, 1.0])
# 应用 Tanh 激活函数
tanh_output = torch.tanh(x)
print(tanh_output)  # 输出：tensor([-0.7616,  0.0000,  0.7616])
