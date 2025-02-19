# -*- coding: utf-8 -*-
"""
@File    : 8.2.文本预处理.py
@Time    : 2025/2/18 15:25
@Desc    : 
"""
import d2l

lines = d2l.read_time_machine()
# print(f"# 文本总行数：{len(lines)}")
# print(lines[0])
# print(lines[10])

tokens = d2l.tokenize(lines)
# for i in range(11):
#     print(tokens[i])

# 构建词表
vocad = d2l.Vocab(tokens)
# print(list(vocad.token_to_idx.items())[:10])


# for i in [0, 10]:
#     print("文本:", tokens[i])
#     print("索引:", vocad[tokens[i]])

corpus, vocab = d2l.load_corpus_time_machine()
print(len(corpus), len(vocab))
