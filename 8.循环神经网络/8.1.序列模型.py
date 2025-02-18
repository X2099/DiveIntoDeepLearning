# -*- coding: utf-8 -*-
"""
@File    : 8.1.序列模型.py
@Time    : 2025/2/17 16:30
@Desc    : 
"""
import torch

import d2l

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, size=(T,))

# d2l.plot(time, x, 'time', 'x', xlim=[1, 1000], figsize=(6, 5.5))

tau = 4
features = torch.zeros((T - tau, tau))
print(features.shape)  # torch.Size([996, 4])

for i in range(tau):
    features[:, i] = x[i:T - tau + i]

labels = x[tau:].reshape((-1, 1))
print(labels.shape)  # torch.Size([996, 1])

from torch import nn


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(
        nn.Linear(4, 18),
        nn.ReLU(),
        nn.Linear(18, 1)
    )
    net.apply(init_weights)
    return net


loss = nn.MSELoss(reduction='none')


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


batch_size, train_n = 16, 600
train_iter = d2l.load_array((features[:train_n], labels[:train_n]),
                            batch_size=batch_size, is_train=True)
net = get_net()
train(net, train_iter, loss, 5, 0.01)

onestep_preds = net(features)
# print(onestep_preds.shape)  # torch.Size([996, 1])
#
# d2l.plot(X=[time, time[tau:]], Y=[x.detach().numpy(), onestep_preds.detach().numpy()],
#          xlabel='time', ylabel='x', legend=['data', '1-step preds'],
#          xlim=[1, 1000], figsize=(6, 5.5))

multistep_preds = torch.zeros(T)
multistep_preds[:train_n + tau] = x[:train_n + tau]
for i in range(train_n + tau, T):
    multistep_preds[i] = net(multistep_preds[i - tau:i])

d2l.plot(
    [time, time[tau:], time[train_n + tau:]],
    [x.detach().numpy(), onestep_preds.detach().numpy(), multistep_preds[train_n + tau:].detach().numpy()],
    xlabel='time', ylabel='x', legend=['data', '1-step preds', 'multi-step preds'],
    xlim=[1, 1000], figsize=(6, 5.5)
)
