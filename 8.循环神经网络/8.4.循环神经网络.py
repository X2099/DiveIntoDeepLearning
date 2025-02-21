# -*- coding: utf-8 -*-
"""
@File    : 8.4.循环神经网络.py
@Time    : 2025/2/21 14:20
@Desc    : 
"""
import torch

torch.manual_seed(49)

# 定义输入矩阵和权重矩阵
X = torch.normal(0, 1, (3, 1))
W_xh = torch.normal(0, 1, (1, 4))  # 输入到隐状态的权重
H = torch.normal(0, 1, (3, 4))  # 隐状态矩阵
W_hh = torch.normal(0, 1, (4, 4))  # 隐状态到隐状态的权重
b_h = torch.normal(0, 1, (1, 4))  # 偏置项

# 更新隐状态
h_t = torch.sigmoid(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
print(h_t)

W_ho = torch.normal(0, 1, (4, 2))  # 隐状态到输出的权重
b_o = torch.normal(0, 1, (1, 2))  # 输出层的偏置项

# 计算输出
y_t = torch.matmul(h_t, W_ho) + b_o
print(y_t)

# 假设我们已经有一个字符编码的输入序列
input_sequence = 'machine'
# 把字符序列转化为向量表示
input_sequence = torch.arange(len(input_sequence), dtype=torch.float32).reshape(-1, 1)

hidden_state = torch.zeros((1, 4))  # 初始化隐状态为零
# 依次处理每个时间步
for t in range(len(input_sequence)):
    input_char = input_sequence[t]
    hidden_state = torch.sigmoid(torch.matmul(input_char, W_xh) + torch.matmul(hidden_state, W_hh) + b_h)
    output_char = torch.matmul(hidden_state, W_ho) + b_o
    print(output_char)

    # 假设模型输出的概率分布是output_probs
    output_probs = torch.softmax(output_char, dim=1)
    # 目标字符是时间步t+1的真实字符
    target = torch.tensor([h])
    # 计算交叉熵损失
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output_probs, target)
