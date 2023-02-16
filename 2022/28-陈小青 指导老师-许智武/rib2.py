# -*- coding = utf-8 -*-
# @Time : 2022-01-23 9:32
# @Author : XiaoJing
# @File : rib.py
# @Software : PyCharm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Rib2(nn.Module):
    def __init__(self, emb_size, item_num, seq_len, dropout_rate=0.5, use_cuda=True):
        super(Rib2, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.input_size = emb_size
        self.output_size = self.hidden_size
        self.seq_len = seq_len
        self.item_num = int(item_num)
        self.dropout_rate = dropout_rate
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.behavior_num = 2
        self.num_layers = 1
        # 初始化参数
        self.item_embeddings = nn.Embedding(self.item_num + 1, self.emb_size, padding_idx=self.item_num)
        self.behavior_embeddings = nn.Embedding(self.behavior_num + 1, self.emb_size, padding_idx=self.behavior_num)
        # # 初始化权重
        # self.item_embeddings.weight.data.normal_(0, 0.01)
        # self.behavior_embeddings.weight.data.normal_(0, 0.01)

        self.gru = nn.GRU(self.input_size * 2, self.hidden_size, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.output_size, self.hidden_size)
        self.fc2 = nn.Linear(self.output_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.h2o = nn.Linear(self.hidden_size, self.item_num)

        self.gru2 = nn.GRU(self.input_size * 2, self.hidden_size, self.num_layers, batch_first=True)
        


    def forward(self, inputs, h0,h1):
        item_seq, behavior_seq, len_seq = inputs
        item_emb = self.item_embeddings(item_seq)  # B*seq_len*emb_size
        behavior_emb = self.behavior_embeddings(behavior_seq)  # B*seq_len*emb_size
        new_input_emb = torch.cat([item_emb, behavior_emb], 2)


        # 处理不定长序列问题
        # new_input_emb = new_input_emb.permute(1, 0, 2)
        packed_input = pack_padded_sequence(new_input_emb, len_seq, enforce_sorted=False, batch_first=True)
        packed_input = packed_input.to(self.device)
        gru_out, states_hidden = self.gru(packed_input, h0)
        gru_out2, states_hidden2 = self.gru2(packed_input, h1)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True, total_length=self.seq_len)  # 128*50*64
        gru_out2, _ = pad_packed_sequence(gru_out2, batch_first=True, total_length=self.seq_len)  # 128*50*64
        gru_out = torch.add(gru_out , gru_out2)/2


        # 注意力层
        att_net = self.tanh(gru_out)  # 128*50*64
        att_net = self.fc2(att_net)
        mask = (~torch.eq(item_seq, self.item_num)).unsqueeze(-1)  # 128*50*1
        paddings = torch.ones_like(att_net) * (-2 ** 32 + 1)  # 128*50*1
        att_net = torch.where(mask, att_net, paddings)  # 128*50*1
        att_net = self.softmax(att_net)  # 128*50*1

        # 输出层
        final_state = torch.sum(gru_out * att_net, 1)
        # print(final_state.size())
        output = self.dropout(final_state)
        output = self.h2o(output)
        # print(output.size())
        return output

    def init_hidden(self, b):
        h0 = torch.zeros(self.num_layers, b, self.hidden_size).to(self.device)
        return h0

