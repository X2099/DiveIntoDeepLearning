# -*- coding: utf-8 -*-
"""
@File    : 10.4.Bahdanau注意力.py
@Time    : 2025/3/20 15:58
@Desc    : 
"""
import torch

import d2l

encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.eval()

X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size, num_steps)
output, state = encoder(X)
print(output.shape)  # torch.Size([7, 4, 16]) (num_steps, batch_size, num_hiddens)
print(state.shape)  # torch.Size([2, 4, 16]) (num_layers, batch_size, num_hiddens)

decoder = d2l.Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                      num_layers=2)
decoder.eval()

outputs, hidden_state, enc_valid_lens = decoder.init_state((output, state), None)

print(outputs.shape)  # torch.Size([4, 7, 16]) (batch_size, num_steps, num_hiddens)
print(hidden_state.shape)  # torch.Size([2, 4, 16]) (num_layers, batch_size, num_hiddens)
print(enc_valid_lens)  # None

output, state = decoder(X, (outputs, hidden_state, enc_valid_lens))

print(output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)
# output.shape : torch.Size([4, 7, 10]) (batch_size, num_steps, vocab_size)
# len(state) : 3  [enc_outputs, hidden_state, enc_valid_lens]
# state[0]: enc_outputs
# state[0].shape <==> enc_outputs.shape : torch.Size([4, 7, 16]) (batch_size, num_steps, num_hiddens)
# len(state[1]) <==> len(hidden_state) <==> num_layers : 2
# state[1][0].shape <==> hidden_state[0].shape : torch.Size([4, 16]) (batch_size, num_hiddens)
# torch.Size([4, 7, 10]) 3 torch.Size([4, 7, 16]) 2 torch.Size([4, 16])

# ---------------------------------------------------------------------------------------------

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = d2l.Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
