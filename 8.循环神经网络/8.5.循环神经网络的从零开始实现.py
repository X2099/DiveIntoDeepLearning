# -*- coding: utf-8 -*-
"""
@File    : 8.5.循环神经网络的从零开始实现.py
@Time    : 2025/2/25 17:22
@Desc    : 
"""
import torch
from torch.nn import functional as F

import d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 词表大小假设为28
X = torch.arange(10).reshape((2, 5))
print(X.T)
"""
tensor([[0, 5],
        [1, 6],
        [2, 7],
        [3, 8],
        [4, 9]])
"""
print(F.one_hot(X.T, 28).shape)


def get_params(vocab_size, num_hiddens, device):
    """
    初始化循环神经网络（RNN）模型的参数。

    参数:
    vocab_size (int): 词汇表大小（即输入和输出的维度）。
    num_hiddens (int): 隐藏层的维度。
    device (torch.device): 设备（如CPU或GPU）。

    返回:
    list: 包含所有模型参数的列表，包含权重矩阵和偏置。
    """
    num_inputs = num_outputs = vocab_size  # 输入和输出的维度等于词汇表大小

    def normal(shape):
        """生成正态分布随机数，初始化参数"""
        return torch.randn(size=shape, device=device) * 0.01  # 标准差为0.01

    # 初始化参数
    W_xh = normal((num_inputs, num_hiddens))  # 输入到隐藏层的权重
    W_hh = normal((num_hiddens, num_hiddens))  # 隐藏层到隐藏层的权重
    b_h = torch.zeros(num_hiddens, device=device)  # 隐藏层的偏置
    W_hq = normal((num_hiddens, num_outputs))  # 隐藏层到输出层的权重
    b_q = torch.zeros(num_outputs, device=device)  # 输出层的偏置

    # 将所有参数放入一个列表
    params = [W_xh, W_hh, b_h, W_hq, b_q]

    # 设置所有参数为可训练
    for param in params:
        param.requires_grad_(True)

    return params


def init_rnn_state(batch_size, num_hiddens, device):
    """
    初始化RNN的隐状态，返回初始的隐藏状态和细胞状态（如果是LSTM的话）。

    参数:
    batch_size (int): 输入数据的批量大小。
    num_hiddens (int): 隐藏层的维度。
    device (torch.device): 设备（如CPU或GPU）。

    返回:
    tuple: 包含隐状态和细胞状态（LSTM）的元组。
    """
    # 对于RNN，初始化隐藏状态为零
    h = torch.zeros(batch_size, num_hiddens, device=device)
    return h,


def rnn(inputs, state, params):
    """
    实现RNN的前向计算过程。

    参数:
    inputs (Tensor): 输入序列的张量，形状为(时间步数, 批量大小, 输入维度)。
    state (tuple): 初始隐状态（对于LSTM包含隐状态和细胞状态）。
    params (list): 模型的所有参数，包含权重和偏置。

    返回:
    tuple: 包含输出和最终隐状态的元组。
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []  # 用来存储每个时间步的输出
    for X in inputs:
        # 当前时间步的隐状态更新公式
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)  # 隐状态的计算
        Y = torch.matmul(H, W_hq) + b_q  # 输出的计算
        outputs.append(Y)  # 保存当前时间步的输出

    return torch.cat(outputs, dim=0), (H,)  # 拼接所有时间步的输出并返回，更新的隐状态


num_hiddens = 512
net = d2l.RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(),
                          get_params, init_rnn_state, rnn)
# state = net.begin_state(X.shape[0], d2l.try_gpu())
# Y, new_state = net(X.to(d2l.try_gpu()), state)
# print(Y.shape, len(new_state), new_state[0].shape)
#
# print(d2l.predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))

num_epochs, lr = 500, 1

d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())

net = d2l.RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                          init_rnn_state, rnn)
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
              use_random_iter=True)
