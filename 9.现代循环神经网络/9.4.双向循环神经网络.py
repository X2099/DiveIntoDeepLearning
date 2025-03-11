# -*- coding: utf-8 -*-
"""
@File    : 9.4.双向循环神经网络.py
@Time    : 2025/3/11 14:49
@Desc    : 
"""
import torch
from torch import nn

import d2l

# 加载数据
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size

# bidirectional=True 表示双向 LSTM（Bidirectional LSTM，BiLSTM）
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, vocab_size)
model = model.to(device)

# 训练模型
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
