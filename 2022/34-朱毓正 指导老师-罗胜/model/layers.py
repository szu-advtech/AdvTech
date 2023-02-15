from model.layers_sub import *


class LstmStatefulLayer(nn.Module):
    def __init__(self, d_input, config, contain_ln=True, bi_direction=False, hidden_state=None):
        super().__init__()
        self.config = config

        self.lstm_cell = nn.LSTMCell(d_input, self.config.d_hidden)
        self.bi_lstm_cell = nn.LSTMCell(d_input, self.config.d_hidden)

        self.hidden_state = hidden_state
        self.bi_hidden_state = None
        self.hidden_reuse_overlap = None
        self._hidden_layer = None

        self.dropout = nn.Dropout(config.dp_ratio)
        self.layer_norm = nn.LayerNorm(config.d_hidden)

        self.dropout_bi = nn.Dropout(config.dp_ratio)
        self.layer_norm_bi = nn.LayerNorm(config.d_hidden)

        self.contain_ln = contain_ln
        self.bi_direction = bi_direction

        self.dropout_connect = nn.Dropout(0.1)
        if bi_direction:
            self.bi_decay = nn.Parameter(torch.arange(1, 0.2, -(0.8 / 30)))
            self.out = nn.Linear(config.d_hidden * 2, config.d_hidden, bias=False)

    def init_hidden(self, batch_size):
        self.hidden_state = (torch.zeros(batch_size, self.config.d_hidden).type(self.config.type_float),
                             torch.zeros(batch_size, self.config.d_hidden).type(self.config.type_float))
        self.bi_hidden_state = (torch.zeros(batch_size, self.config.d_hidden).type(self.config.type_float),
                                torch.zeros(batch_size, self.config.d_hidden).type(self.config.type_float))

    def reuse_hidden(self):
        # self.init_hidden(self.hidden_state[0].size(0))
        batch_size = self.hidden_state[0].size(0)
        self.hidden_state = (self.hidden_state[0].clone().detach(),
                             self.hidden_state[1].clone().detach())
        self.bi_hidden_state = (torch.zeros(batch_size, self.config.d_hidden).type(self.config.type_float),
                                torch.zeros(batch_size, self.config.d_hidden).type(self.config.type_float))

    def reuse_hidden_overlap(self):
        self.hidden_state = (self.hidden_reuse_overlap[0].clone().detach(),
                             self.hidden_reuse_overlap[1].clone().detach())

    def get_hidden_layer(self):
        return self._hidden_layer.clone().detach()

    def clear_hidden(self):
        self.hidden_state = None
        self.bi_hidden_state = None

    def forward(self, inputs):
        batch_size = inputs.size(0)
        win_len = inputs.size(1)

        if self.hidden_state is None:
            self.init_hidden(batch_size)
        elif self.config.isOverlap is False:
            self.reuse_hidden()
        else:
            self.reuse_hidden_overlap()

        outputs_hidden = []
        # enumerating through lstm
        for t in range(win_len):
            if self.config.is_reverse:
                t = win_len - 1 - t
            input_t = inputs[:, t, :]
            # self.hidden_state = self.lstm_cell(input_t, (self.dropout_connect(self.hidden_state[0]), self.hidden_state[1]))
            self.hidden_state = self.lstm_cell(input_t, self.hidden_state)
            # drop out
            # self.hidden_state = (self.dropout(self.hidden_state[0]), self.hidden_state[1])
            # if overlapping, re-use hidden state from middle
            if (self.config.isOverlap is True) and t == (win_len / 2) - 1:
                self.hidden_reuse_overlap = self.hidden_state
            # add layer-norm
            if self.contain_ln:
                # outputs_hidden.append(self.layer_norm(self.hidden_state[0]))
                outputs_hidden.append(self.layer_norm(self.dropout(self.hidden_state[0])))
            else:
                outputs_hidden.append(self.dropout(self.hidden_state[0]))

        outputs_hidden = torch.stack(outputs_hidden, 1)

        if self.bi_direction and not self.config.is_reverse:
            # self.bi_hidden_state = self.hidden_state
            outputs_bi_hidden = []
            for t in range(win_len - 1, -1, -1):
                input_t = inputs[:, t, :]
                self.bi_hidden_state = self.bi_lstm_cell(input_t, self.bi_hidden_state)

                # self.bi_hidden_state = (self.dropout_bi(self.bi_hidden_state[0]), self.bi_hidden_state[1])

                outputs_bi_hidden.append(self.layer_norm_bi(self.dropout(self.bi_hidden_state[0])))
            outputs_bi_hidden = torch.stack(outputs_bi_hidden, 1)
            # outputs_bi_hidden = outputs_bi_hidden*self.bi_decay[None,:,None]

            outputs_hidden = torch.cat((outputs_hidden, outputs_bi_hidden), 2)

        if self.config.isOverlap:
            self._hidden_layer = outputs_hidden[:, :(win_len // 2) - 1, :]
        else:
            # self._hidden_layer = outputs_hidden[:, (win_len // 2):, :]
            self._hidden_layer = outputs_hidden
        return outputs_hidden


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, pe=True):
        super(SelfAttentionLayer, self).__init__()
        self.d_model = d_model
        self.is_pe = pe
        self.pe = PositionalEncoding_old(d_model)
        self.self_att = MultiHeadedAttention(d_model, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.feedForward = PositionwiseFeedForward(d_model, d_model * 4)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x_or = x

        x = self.pe(x)
        if self.is_pe:
            x_or = x
        att_outputs = self.self_att(x, x, x)
        f_outputs = self.feedForward(self.dropout_1(att_outputs))
        return x_or + self.dropout(f_outputs)


# class AdversaryLayer(nn.Module):
#     def __init__(self, d_model, dropout=0.1, pe=True):

## Channel wise attention(CA) layer
class CALayer(nn.Module):
    def __init__(self, config):
        super(CALayer, self).__init__()
        self.config = config
        # BxSxA -> Bx1xA
        self.avg_pool = nn.AdaptiveAvgPool2d((1, self.config.input_dim))
        # Bx1xA -> Bx1xH Dimension reduction
        self.fc_1 = nn.Linear(in_features=self.config.input_dim, out_features=self.config.d_reduction, bias=True)
        # Bx1xH -> Bx1xA Dimension increasing
        self.fc_2 = nn.Linear(in_features=self.config.d_reduction, out_features=self.config.input_dim, bias=True)
        self.act_t = nn.Tanh()
        self.act_s = nn.Softmax(dim=1)

    def forward(self, x):
        # Transfer to channel wise pooling
        # Pooling function BxSxA -> Bx1xA
        out = self.avg_pool(x)
        # Bx1xA -> Bx1xH
        out = self.act_t(self.fc_1(out))
        # Bx1xH -> Bx1xA
        out = self.act_s(self.fc_2(out))

        return x * out
