# -*- coding: utf-8 -*-
"""
@File    : 2.5.自动微分.py
@Time    : 2024/10/25 9:57
@Desc    : 
"""

import torch


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


torch.manual_seed(49)

a = torch.randn(size=(), requires_grad=True)
print(a)  # tensor(-2.0157, requires_grad=True)
d = f(a)
print(d)  # tensor(-103203.8359, grad_fn=<MulBackward0>)
d.backward()
print(a.grad)  # tensor(51200.)

print(a.grad == d / a)  # tensor(True)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # 向量 x
#
# y = x * x
# print(y)  # tensor([1., 4., 9.], grad_fn=<MulBackward0>)
#
# u = y.clone()  # 这里并没有分离出 u
#
# # 现在我们用未分离的 𝑢 来计算 𝑧，得到：
# z = y * x
# print(z)  # tensor([ 1.,  8., 27.], grad_fn=<MulBackward0>)
#
# z.sum().backward()
# print(x.grad)  # tensor([ 3., 12., 27.])

# y.sum().backward()
# print(x.grad)  # tensor([0., 2., 4., 6.])

y = 2 * torch.dot(x, x)
# print(y)  # tensor(28., grad_fn=<MulBackward0>)
#
y.backward()
print(x.grad)  # tensor([ 0.,  4.,  8., 12.])
#
# print(x.grad == 4 * x)  # tensor([True, True, True, True])
