# -*- coding: utf-8 -*-
"""
@File    : 9.7.序列到序列学习seq2seq.py
@Time    : 2025/3/13 17:02
@Desc    : 
"""
import torch

import d2l

# encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8,
#                              num_hiddens=16, num_layers=2)
# encoder.eval()
# X = torch.zeros((4, 7), dtype=torch.long)
# output, state = encoder(X)
# print(output.shape)  # 输出：torch.Size([7, 4, 16])
# print(state.shape)  # 输出：torch.Size([2, 4, 16])
#
# decoder = d2l.Seq2SeqDecoder(vocab_size=10, embed_size=8,
#                              num_hiddens=16, num_layers=2)
# decoder.eval()
# state = decoder.init_state(encoder(X))
# output, state = decoder(X, state)
# print(output.shape)  # 输出：torch.Size([4, 7, 10])
# print(state.shape)  # 输出：torch.Size([2, 4, 16])


# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(d2l.sequence_mask(X, torch.tensor([1, 2])))

# loss = d2l.MaskedSoftmaxCELoss()
# print(loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
#            torch.tensor([4, 2, 0])))

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(vocab_size=len(src_vocab), embed_size=embed_size,
                             num_hiddens=num_hiddens, num_layers=num_layers)
decoder = d2l.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {d2l.bleu(translation, fra, k=2):.3f}')