# -*- coding: utf-8 -*-
"""
@File    : 10.7.Transformer.py
@Time    : 2025/3/27 14:06
@Desc    : 
"""
import torch
from torch import nn

import d2l

# num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
# lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
# ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
# key_size, query_size, value_size = 32, 32, 32
# norm_shape = [32]
#
# train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
#
# encoder = d2l.TransformerEncoder(
#     len(src_vocab), key_size, query_size, value_size, num_hiddens,
#     norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
#     num_layers, dropout)
# decoder = d2l.TransformerDecoder(
#     len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
#     norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
#     num_layers, dropout)
# net = d2l.EncoderDecoder(encoder, decoder)
# d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

# num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 64
# lr, num_epochs, device = 0.01, 100, d2l.try_gpu()
# ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
# key_size, query_size, value_size = 32, 32, 32
# norm_shape = [32]
#
# train_iter, src_vocab, tgt_vocab = d2l.load_data_zh_en(batch_size, num_steps)
#
# encoder = d2l.TransformerEncoder(
#     len(src_vocab), key_size, query_size, value_size, num_hiddens,
#     norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
#     num_layers, dropout)
# decoder = d2l.TransformerDecoder(
#     len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
#     norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
#     num_layers, dropout)
# net = d2l.EncoderDecoder(encoder, decoder)
# d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# encoder = d2l.TransformerEncoder(vocab_size=200, key_size=24, query_size=24, value_size=24,
#                                  num_hiddens=24, norm_shape=[100, 24], ffn_num_input=24, ffn_num_hiddens=48,
#                                  num_heads=8, num_layers=2, dropout=0.5)
# encoder.eval()
# valid_lens = torch.tensor([3, 2])
# print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)
# # 输出：torch.Size([2, 100, 24])


# X = torch.ones((2, 100, 24))
# valid_lens = torch.tensor([3, 2])
# encoder_blk = d2l.EncoderBlock(key_size=24, query_size=24, value_size=24, num_hiddens=24,
#                                norm_shape=[100, 24], ffn_num_input=24, ffn_num_hiddens=48,
#                                num_heads=8, dropout=0.5)
# # encoder_blk.eval()
# # print(encoder_blk(X, valid_lens).shape)  # 输出：torch.Size([2, 100, 24])
#
# decoder_blk = d2l.DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
# decoder_blk.eval()
# X = torch.ones((2, 100, 24))
# state = [encoder_blk(X, valid_lens), valid_lens, [None]]
# print(decoder_blk(X, state)[0].shape)  # 输出：torch.Size([2, 100, 24])

add_norm = d2l.AddNorm(normalized_shape=[3, 4], dropout=0.5)
add_norm.eval()
X = torch.ones((2, 3, 4))
Y = torch.ones((2, 3, 4))

print(add_norm(X, Y))
print(add_norm(X, Y).shape)  # 输出：torch.Size([2, 3, 4])

# ln = nn.LayerNorm(2)
# bn = nn.BatchNorm1d(2)
#
# X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
#
# # 在训练模式下计算X的均值和方差
# print('layer norm:', ln(X), '\nbatch norm:', bn(X))

# ffn = d2l.PositionWiseFFN(4, 4, 8)
# ffn.eval()
#
# intput = torch.ones((2, 3, 4))
# output = ffn(intput)
#
# print(output.shape)  # 输出：torch.Size([2, 3, 8])
#
# print(output[0])
