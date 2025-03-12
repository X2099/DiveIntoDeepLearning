# -*- coding: utf-8 -*-
"""
@File    : 9.5.机器翻译与数据集.py
@Time    : 2025/3/11 16:55
@Desc    : 
"""
import torch
import d2l

# raw_text = d2l.read_data_nmt()
# text = d2l.preprocess_nmt(raw_text)
#
# source, target = d2l.tokenize_nmt(text)

# d2l.show_list_len_pair_hist(['source', 'target'], '# tokens per sequence', 'count', source, target)

# src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
# print(len(src_vocab))  # 输出：10012
# print(src_vocab.token_to_idx)

# print(d2l.truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

# 读取一个小批量数据示例
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break
