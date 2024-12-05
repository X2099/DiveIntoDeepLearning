# -*- coding: utf-8 -*-
"""
@File    : 2.4.微积分.py
@Time    : 2024/10/23 19:06
@Desc    : 
"""

import numpy as np
import d2l


def f(x):
    return 3 * x ** 2 - 4 * x


x = np.arange(0, 3, 0.1)
d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line（切线） (x=1)'])

#
#
# def numerical_lim(f, x, h):
#     return (f(x + h) - f(x)) / h
#
#
# h = 0.1
# for i in range(5):
#     print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
#     h *= 0.1

"""
h=0.10000, numerical limit=2.30000
h=0.01000, numerical limit=2.03000
h=0.00100, numerical limit=2.00300
h=0.00010, numerical limit=2.00030
h=0.00001, numerical limit=2.00003
"""
