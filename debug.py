# -*- coding: utf-8 -*-
"""
@File    : debug.py
@Time    : 2024/3/8 10:18
@Desc    : 
"""
import torch
from torch import nn


def dropout_layer(X, dropout):
    return X


dropout1, dropout2 = 0.2, 0.5

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256


class Net(nn.Module):

    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super().__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.line1_layer = nn.Linear(num_inputs, num_hiddens1)
        self.line2_layer = nn.Linear(num_hiddens1, num_hiddens2)
        self.line3_layer = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        """
        前向算法
        :return:
        """
        H1 = self.relu(self.line1_layer(X.reshape(-1, self.num_inputs)))
        if self.training:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.line2_layer(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)
        return self.line3_layer(H2)


net = Net(num_inputs, num_inputs, num_hiddens1, num_hiddens2)
