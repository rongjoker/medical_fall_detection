# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/26 14:30
@Auth ： hcb
@File ：lstm_model.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):

    def __init__(self, args):
        super(LSTMClassifier, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.word_embeddings = nn.Embedding(args.vocab_num, args.embedding_dim)
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim, batch_first=True, bidirectional=True)
        self.hidden2label = nn.Linear(args.hidden_dim * 2, args.class_num)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        if self.args.use_cuda:
            return (autograd.Variable(torch.zeros(2, self.args.batch_size, self.hidden_dim)).cuda(),
                    autograd.Variable(torch.zeros(2, self.args.batch_size, self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(2, self.args.batch_size, self.hidden_dim)),
                    autograd.Variable(torch.zeros(2, self.args.batch_size, self.hidden_dim)))

    def forward(self, sentence, lengths=None):
        """"""
        if not lengths:
            self.hidden = self.init_hidden()
            embeds = self.word_embeddings(sentence)
            x = embeds
            lstm_out, self.hidden = self.lstm(x, self.hidden)
            y = self.hidden2label(lstm_out[:,-1])  # 分类选择所有行的最后一个隐层
            log_probs = F.log_softmax(y)
        else:
            self.hidden = self.init_hidden()
            embeds = self.word_embeddings(sentence)
            x = embeds
            x_pack = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
            lstm_out, self.hidden = self.lstm(x_pack, self.hidden)
            lstm_out, output_lens  = pad_packed_sequence(lstm_out, batch_first=True)
            y = self.hidden2label(lstm_out[:,-1])  # 分类选择所有行的最后一个隐层
            log_probs = F.log_softmax(y)

        return log_probs
