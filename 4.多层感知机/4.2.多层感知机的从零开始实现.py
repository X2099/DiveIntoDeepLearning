# -*- coding: utf-8 -*-
"""
@File    : 4.2.多层感知机的从零开始实现.py
@Time    : 2024/11/15 15:22
@Desc    : 
"""
import torch
from torch import nn
import d2l

num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 初始化权重和偏置
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

# 将参数打包到一起，方便后续更新
params = [W1, b1, W2, b2]


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    X = X.reshape((-1, num_inputs))  # 将图像展平
    H = relu(torch.matmul(X, W1) + b1)  # 隐藏层的计算
    y = torch.matmul(H, W2) + b2  # 输出层的计算
    return y


loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr, batch_size = 10, 0.1, 256
updater = torch.optim.SGD(params, lr=lr)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

metric = d2l.Accumulator(3)
animator = d2l.Animator(xlabel='轮数', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'test acc'], title="多层感知机模型训练效果图")
# 训练模型
d2l.train_softmax(net, train_iter, test_iter, loss,
                  num_epochs, updater, batch_size, animator)
# 可视化优化过程
animator.show()

d2l.predict_ch3(net, test_iter)
