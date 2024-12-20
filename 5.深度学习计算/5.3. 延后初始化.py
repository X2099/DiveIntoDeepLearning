# -*- coding: utf-8 -*-
"""
@File    : 5.3. 延后初始化.py
@Time    : 2024/12/18 16:21
@Desc    : 
"""
import torch
from torch import nn

# 使用 LazyLinear 定义一个两层的网络
net = nn.Sequential(
    nn.LazyLinear(256),  # 第一层：输出维度256，输入维度将延后初始化
    nn.ReLU(),  # 激活函数：ReLU
    nn.LazyLinear(10)  # 第二层：输出维度10
)

# 查看每一层（注意：此时权重还未初始化）
for i, layer in enumerate(net):
    if isinstance(layer, nn.LazyLinear):
        print(f"第 {i + 1} 层（在前向传播之前）: {layer}")

# 生成一个随机输入数据
X = torch.randn(2, 20)  # 随机生成一个2x20的输入数据

# 将数据通过网络进行一次前向传播，触发参数初始化
output = net(X)

# 查看每一层的权重形状
for i, layer in enumerate(net):
    if isinstance(layer, nn.Linear):
        print(f"第 {i + 1} 层 {layer} 的权重形状（在前向传播之后）: {layer.weight.shape}")
