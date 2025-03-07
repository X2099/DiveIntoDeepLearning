# -*- coding: utf-8 -*-
"""
@File    : 9.2.长短期记忆网络LSTM.py
@Time    : 2025/3/7 14:56
@Desc    : 
"""
import torch
from torch import nn

import d2l

batch_size, num_steps = 32, 25
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.rand(shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_lstm_state(batch_size, num_hiddens, device):
    """初始化长短期记忆网络的隐状态"""
    return (torch.zeros(size=(batch_size, num_hiddens), device=device),
            torch.zeros(size=(batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    H, C = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)  # 输入门
        F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)  # 遗忘门
        O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)  # 输出门
        C_tilda = torch.tanh(X @ W_xc + H @ W_hc + b_c)  # 候选记忆元
        C = F * C + I * C_tilda  # 记忆元
        H = O * torch.tanh(C)  # 隐状态
        Y = H @ W_hq + b_q  # 输出
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(vocab_size, num_hiddens, device, get_lstm_params, init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, vocab_size)
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
