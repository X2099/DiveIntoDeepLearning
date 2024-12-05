# -*- coding: utf-8 -*-
"""
@File    : 4.3.多层感知机的简洁实现.py
@Time    : 2024/11/18 11:05
@Desc    : 
"""
import torch
from torch import nn
import d2l

# 定义模型
net = nn.Sequential(
    nn.Flatten(),  # 展平输入
    nn.Linear(784, 256),  # 隐藏层
    nn.ReLU(),  # 激活函数
    nn.Linear(256, 10)  # 输出层
)


# 初始化参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter,
              loss, num_epochs, trainer, batch_size)
