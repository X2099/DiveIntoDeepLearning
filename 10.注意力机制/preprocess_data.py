# -*- coding: utf-8 -*-
"""
@File    : preprocess_data.py
@Time    : 2025/4/9 10:47
@Desc    : 数据预处理
"""
import re
from itertools import islice
import jieba
import torch
from torch import nn

import d2l


def read_data():
    data_zh_path = 'data/training-parallel-nc-v13/news-commentary-v13.zh-en.zh'
    data_en_path = 'data/training-parallel-nc-v13/news-commentary-v13.zh-en.en'
    with open(data_zh_path, 'r', encoding='utf-8') as f:
        lines_zh = list(islice(f, 10000))
    with open(data_en_path, 'r', encoding='utf-8') as f:
        lines_en = list(islice(f, 10000))
    tokens_zh = [jieba.lcut(line.strip().replace(' ', '')) for line in lines_zh]
    tokens_en = [re.findall(r"\w+|[^\w\s]", line) for line in lines_en]
    total_tokens = 0

    for line in tokens_zh:
        total_tokens += len(line)
    print(total_tokens / len(tokens_zh))


def load_data_zh_en(batch_size, num_steps):
    """返回中英翻译数据集的迭代器和词表"""
    data_zh_path = 'data/training-parallel-nc-v13/news-commentary-v13.zh-en.zh'
    data_en_path = 'data/training-parallel-nc-v13/news-commentary-v13.zh-en.en'
    with open(data_zh_path, 'r', encoding='utf-8') as f:
        lines_zh = list(islice(f, 10000))
    with open(data_en_path, 'r', encoding='utf-8') as f:
        lines_en = list(islice(f, 10000))

    tokens_zh = [jieba.lcut(line.strip().replace(' ', '')) for line in lines_zh]
    tokens_en = [re.findall(r"\w+|[^\w\s]", line) for line in lines_en]
    # tokens_zh = [list(line.strip().replace(' ', '')) for line in lines_zh]
    # tokens_en = [list(line.strip()) for line in lines_en]
    vocab_zh = d2l.Vocab(tokens_zh, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    vocab_en = d2l.Vocab(tokens_en, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    print(len(vocab_zh))
    print(len(vocab_en))

    src_array, src_valid_len = d2l.build_array_nmt(tokens_zh, vocab_zh, num_steps)
    tgt_array, tgt_valid_len = d2l.build_array_nmt(tokens_en, vocab_en, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, vocab_zh, vocab_en


def debug():
    model = nn.Linear(3, 10)
    X = torch.tensor([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
    print('X.shape:', X.shape)

    for parm in model.parameters():
        print(parm.T)
    Y = model(X)
    print('Y:', Y)
    print('Y.shape:', Y.shape)


if __name__ == '__main__':
    batch_size, num_steps = 64, 32
    train_iter, src_vocab, tgt_vocab = load_data_zh_en(batch_size, num_steps)
    embedding = nn.Embedding(len(src_vocab), 10)
    print('embedding.weight.shape:', embedding.weight.shape)
    print(embedding.weight[:10])
    for batch in train_iter:
        X, X_valid_len, Y, Y_valid_len = batch
        print(X.shape)
        print(X)
        X_embed = embedding(X)
        print('X_embed shape:', X_embed.shape)
        print(X_valid_len.shape)
        print(X_valid_len)

        # print(Y.shape)
        # print(Y)
        # print(Y_valid_len.shape)
        # print(Y_valid_len)
        break
    # read_data()
    # debug()
