# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:35:23 2021

@author: sarvesh
"""

import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable

import numpy as np


class LSTM_attentModel(torch.nn.Module):
    def __init__(self, vocab_size, output_size, hidden_size, batch_size, wghts, embed_len):
        super(LSTM_attentModel, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embed_len = embed_len
        self.embedded_words = nn.Embedding(vocab_size, embed_len)
        self.embedded_words_wgts = nn.Parameter(wghts, requires_grad=False)
        self.lstm = nn.LSTM(embed_len, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def attention(self, model_op, f_state):
        hidden = f_state.squeeze(0)
        wt_att = torch.bmm(model_op, hidden.unsqueeze(2)).squeeze(2)
        soft_wt_attn = functional.softmax(wt_att, 1)
        res = torch.bmm(model_op.transpose(1, 2),
                        soft_wt_attn.unsqeeze(2)).squeeze(2)

        return res

    def lstm_layer(self, input_sent, batch_size=None):
        inp = self.embedded_words(input_sent)
        inp = inp.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(
                1, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(
                1, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        output, (final_hidden_size, final_cell_state) = self.lstm(
            input, (h_0, c_0))
        output = output.permute(1, 0, 2)
        attn_op = self.attention(output, final_hidden_size)
        labels = self.label(attn_op)

        return labels
