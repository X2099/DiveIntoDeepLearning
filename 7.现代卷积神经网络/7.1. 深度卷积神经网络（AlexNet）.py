# -*- coding: utf-8 -*-
"""
@File    : 7.1. 深度卷积神经网络（AlexNet）.py
@Time    : 2025/1/7 15:20
@Desc    : 
"""
import torch
from torch import nn
import d2l

net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)

# 创建一个示例输入数据并观察每层的输出形状
# X = torch.randn(1, 1, 224, 224)  # 224x224单通道输入
# print(f'X初始值\n', '\tinput shape:\t', X.shape)
# for layer in net:
#     X = layer(X)
#     print(f'{layer}\n', '\toutput shape:\t', X.shape)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
# for X, y in train_iter:
#     print(X.shape)
#     break

lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
