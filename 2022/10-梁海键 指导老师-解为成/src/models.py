import math

import torch
from torch import nn
import torch.nn.functional as F

from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.transformer import TransformerEncoder, MRUEncoderBlock, Linear


class PMRModel(nn.Module):
    def __init__(self, hyp_params):
        """
            Construct a PMR model.
        """
        super(PMRModel, self).__init__()
        # 300， 74， 35
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        # self.d_l, self.d_a, self.d_v, self.embed_dim = 30, 30, 30, 30
        self.d_l, self.d_a, self.d_v, self.embed_dim = 40, 40, 40, 40
        self.num_heads = hyp_params.num_heads  # 5
        self.layers = hyp_params.layers  # 5
        self.attn_dropout = hyp_params.attn_dropout  # 0.1
        self.relu_dropout = hyp_params.relu_dropout  # 0.1
        self.res_dropout = hyp_params.res_dropout  # 0.1
        self.out_dropout = hyp_params.out_dropout  # 0
        self.embed_dropout = hyp_params.embed_dropout  # 0.25
        self.attn_mask = hyp_params.attn_mask  # true

        self.embed_positions = SinusoidalPositionalEmbedding(self.embed_dim)

        # 1. Temporal convolutional layers
        # 因为现在的每个模态数据都是一维向量，所以此时使用1D卷积核
        # self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        # self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        # self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=3, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=5, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=3, padding=0, bias=False)

        # 2. Crossmodal Attentions

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)

        self.layerNorm = nn.LayerNorm(self.embed_dim)
        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)  # The "Add & Norm" part in the paper
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)

        self.MRUblock = MRUEncoderBlock(self.embed_dim,
                                        num_heads=self.num_heads,
                                        attn_dropout=self.attn_dropout,
                                        relu_dropout=self.relu_dropout,
                                        res_dropout=self.res_dropout,
                                        attn_mask=self.attn_mask)

        self.transformer_all = TransformerEncoder(embed_dim=self.embed_dim,
                                                  num_heads=self.num_heads,
                                                  layers=self.layers,
                                                  attn_dropout=self.attn_dropout,
                                                  relu_dropout=self.relu_dropout,  # 0.1
                                                  res_dropout=self.res_dropout,  # 0.1
                                                  embed_dropout=self.embed_dropout,  # 0、25
                                                  attn_mask=self.attn_mask)  # true

        # Projection layers
        output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)
        self.proj1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_layer = nn.Linear(self.embed_dim, output_dim)

    def pre_embed_position(self, x_v, x_l, x_a):
        embed_scale = math.sqrt(self.embed_dim)
        x_v = embed_scale * x_v
        if self.embed_positions is not None:
            x_v += self.embed_positions(x_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
        x_v = F.dropout(x_v, p=self.embed_dropout, training=self.training)

        x_l = embed_scale * x_l
        if self.embed_positions is not None:
            x_l += self.embed_positions(x_l.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
        x_l = F.dropout(x_l, p=self.embed_dropout, training=self.training)

        x_a = embed_scale * x_a
        if self.embed_positions is not None:
            x_a += self.embed_positions(x_a.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
        x_a = F.dropout(x_a, p=self.embed_dropout, training=self.training)

        return x_v, x_l, x_a

    def MUMblock(self, x_v, x_l, x_a):
        posi_0 = x_v.size(0)
        posi_1 = x_v.size(1)
        posi_2 = x_v.size(2)
        embed_dim = x_v.size(0) * x_v.size(2)

        # 先对各个模态特征进行展平处理
        x_v = x_v.view(1, x_v.size(1), embed_dim)
        x_l = x_l.view(1, x_l.size(1), embed_dim)
        x_a = x_a.view(1, x_a.size(1), embed_dim)

        # linear 如果是三维的输入，会将前两维的数据先乘一起，然后在做计算，
        # 这个函数就是改变最后一维，也就是数据的特征维度，通过调整output_size的尺寸来扩张或者是收缩特征。
        linear_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        trans_proj = nn.Linear(embed_dim, 1, bias=False)

        # 计算每种模态的权重，控制通过量，对应于论文中的注意力层
        single_v = trans_proj(torch.tanh(linear_proj(x_v)))
        single_v = F.dropout(single_v, p=self.embed_dropout, training=self.training)
        single_l = trans_proj(torch.tanh(linear_proj(x_l)))
        single_l = F.dropout(single_l, p=self.embed_dropout, training=self.training)
        single_a = trans_proj(torch.tanh(linear_proj(x_a)))
        single_a = F.dropout(single_a, p=self.embed_dropout, training=self.training)

        sum_atten = torch.exp(single_l) + torch.exp(single_v) + torch.exp(single_a)
        atten_v = torch.exp(single_v) / sum_atten
        atten_l = torch.exp(single_l) / sum_atten
        atten_a = torch.exp(single_a) / sum_atten

        # 合并得到全面的共同信息特征
        common_message = atten_a * (x_a.view(posi_0, posi_1, posi_2)) + atten_l * (x_l.view(posi_0, posi_1, posi_2)) + atten_v * (x_v.view(posi_0, posi_1, posi_2))

        # 通过PFF层和LN对其进行处理，得到最终输出
        residual = common_message
        common_message = self.layerNorm(common_message)
        common_message = F.relu(self.fc1(common_message))
        common_message = F.dropout(common_message, p=self.relu_dropout, training=self.training)
        common_message = self.fc2(common_message)
        common_message = F.dropout(common_message, p=self.res_dropout, training=self.training)
        common_message = residual + common_message

        return common_message

    def forward(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = F.dropout(x_a.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = F.dropout(x_v.transpose(1, 2), p=self.embed_dropout, training=self.training)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)  # [seq_len, batch_size, n_features]
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        individual_v, individual_l, individual_a = self.pre_embed_position(proj_x_v, proj_x_l, proj_x_a)

        common_m = torch.cat((individual_v, individual_l, individual_a), dim=0)
        # embed_scale = math.sqrt(self.embed_dim)
        # common_m = embed_scale * common_m

        for layer in range(self.layers):
            # (sequence_length, batch_size, dimension)
            individual_a = self.MRUblock(individual_a, common_m, common_m)
            individual_l = self.MRUblock(individual_l, common_m, common_m)
            individual_v = self.MRUblock(individual_v, common_m, common_m)

            a_t_oc = self.MRUblock(common_m, individual_a, individual_a)
            l_t_oc = self.MRUblock(common_m, individual_l, individual_l)
            v_t_oc = self.MRUblock(common_m, individual_v, individual_v)
            common_m = self.MUMblock(v_t_oc, l_t_oc, a_t_oc)

        combined_f = torch.cat([common_m, individual_l, individual_v, individual_a], dim=0)
        combined_f = self.transformer_all(combined_f)
        if type(combined_f) == tuple:
            h_ls = combined_f[0]
        # {batch_size, feature_dim}
        last_hs = combined_f[-1]  # Take the last output for prediction

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output, last_hs


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        # 300， 74， 35
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        # self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.d_l, self.d_a, self.d_v, self.embed_dim = 40, 40, 40, 40

        # true three
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads  # 5
        self.layers = hyp_params.layers  # 5
        self.attn_dropout = hyp_params.attn_dropout  # 0.1
        self.attn_dropout_a = hyp_params.attn_dropout_a  # 0
        self.attn_dropout_v = hyp_params.attn_dropout_v  # 0
        self.relu_dropout = hyp_params.relu_dropout  # 0.1
        self.res_dropout = hyp_params.res_dropout  # 0.1
        self.out_dropout = hyp_params.out_dropout  # 0
        self.embed_dropout = hyp_params.embed_dropout  # 0.25
        self.attn_mask = hyp_params.attn_mask  # true

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly  # true
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l  # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        # 1
        output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        # 因为现在的每个模态数据都是一维向量，所以此时使用1D卷积核
        # 此时1*1卷积，就相当于是全连接了
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=3, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=3, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=3, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,  # 0.1
                                  res_dropout=self.res_dropout,  # 0.1
                                  embed_dropout=self.embed_dropout,  # 0、25
                                  attn_mask=self.attn_mask)  # true

    def forward(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)  # [seq_len, batch_size, n_features]
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
            # h_ls {seq_len, batch_size, 2 * feature_dim}
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            # h_ls tensor{seq_len, batch_size, 2 * feature_dim}
            h_ls = self.trans_l_mem(h_ls)

            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            # {batch_size, 2 * feature_dim}
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        # last_hs {batch_size, 3 * 2 * feature_dim}
        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output, last_hs
