# -*- coding: utf-8 -*-
"""
@File    : 8.3.语言模型和数据集.py
@Time    : 2025/2/19 16:55
@Desc    : 
"""
from pprint import pprint

import d2l

tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)

# pprint(vocab.token_freqs[:10])


freqs = [freq for token, freq in vocab.token_freqs]
# d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
#          xscale='log', yscale='log')

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
# pprint(bigram_vocab.token_freqs[:10])


trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
# pprint(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]

# d2l.plot([freqs, bigram_freqs, trigram_freqs],
#          xlabel='token: x', ylabel='frequency: n(x)',
#          xscale='log', yscale='log', legend=['unigram', 'bigram', 'trigram'])

# my_seq = list(range(35))
# 打印每次生成的输入（X）和标签（Y）
# for X, Y in d2l.seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
#     print('X:', X, '\nY:', Y)

# for X, Y in d2l.seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
#     print('X:', X, '\nY:', Y)

data_iter, vocab = d2l.load_data_time_machine(batch_size=2, num_steps=5)

print(vocab.token_freqs[:10])
for X, Y in data_iter:
    print('X:', X, '\nY:', Y)
    break
