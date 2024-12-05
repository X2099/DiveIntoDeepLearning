# -*- coding: utf-8 -*-
"""
@File    : 4.8.数值稳定性与模型初始化.py
@Time    : 2024/12/4 16:51
@Desc    : 
"""
import torch
from d2l import plt

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)

y.backward(torch.ones_like(x))

plt.plot(x.detach().numpy(), y.detach().numpy(), label="Sigmoid")
plt.plot(x.detach().numpy(), x.grad.numpy(), label="Gradient")
plt.legend()


# plt.show()

# M = torch.normal(0, 1, size=(4, 4))
# print("初始矩阵：\n", M)
# for i in range(100):
#     M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))
# print("乘100次之后：\n", M)


# torch.manual_seed(42)  # 设置随机种子，确保可重复性
# weights = torch.randn(2, 2)  # 随机初始化一个 2x2 权重矩阵


# print(weights)
# """
# tensor([[0.3367, 0.1288],
#         [0.2345, 0.2303]])
# """


def xavier_init(shape):
    fan_in, fan_out = shape
    limit = torch.sqrt(torch.tensor(6.0) / (fan_in + fan_out))
    return torch.empty(shape).uniform_(-limit, limit)


weights = xavier_init((256, 512))
print(weights.shape)
print(weights)
