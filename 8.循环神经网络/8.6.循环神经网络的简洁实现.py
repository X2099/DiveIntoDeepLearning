# -*- coding: utf-8 -*-
"""
@File    : 8.6.循环神经网络的简洁实现.py
@Time    : 2025/3/5 10:59
@Desc    : 
"""
import torch
from torch import nn
from torch.nn import functional as F

import d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

num_hiddens = 256
rnn_layer = nn.RNN(input_size=len(vocab), hidden_size=num_hiddens)

state = torch.zeros(size=(1, batch_size, num_hiddens))

X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)

# print(Y.shape)  # 输出：torch.Size([35, 32, 256])


device = d2l.try_gpu()
print(device)
print(torch.cuda.is_available())  # 是否能检测到 CUDA 设备
print(torch.__version__)  # PyTorch 版本
print(torch.version.cuda)  # PyTorch 绑定的 CUDA 版本（如果支持 CUDA）
net = d2l.RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
# print(d2l.predict_ch8('time traveller', 10, net, vocab, device))

num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
