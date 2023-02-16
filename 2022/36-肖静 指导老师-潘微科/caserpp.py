import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, emb_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(emb_size))
        self.bias = nn.Parameter(torch.zeros(emb_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FilterLayer(nn.Module):
    def __init__(self, args, seq_len):
        super(FilterLayer, self).__init__()
        # torch.randn(batch, channel, height, width)
        self.complex_weight = nn.Parameter(
            torch.randn(1, seq_len // 2 + 1, args.emb_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.filter_dropout_prob)
        self.LayerNorm = LayerNorm(args.emb_size, eps=1e-12)

    def forward(self, input_emb):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_emb.shape  # (256, 50, 64)
        x = torch.fft.rfft(input_emb, dim=1, norm='ortho')  # (256, 26, 64)
        weight = torch.view_as_complex(self.complex_weight)  # (1, 26, 64)
        x = x * weight  # (256, 26, 64)
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')  # (256, 50, 64)
        filter_tensor = self.out_dropout(sequence_emb_fft)
        filter_tensor = self.LayerNorm(filter_tensor + input_emb)

        return filter_tensor


class Encoder(nn.Module):
    def __init__(self, args, seq_len):
        super(Encoder, self).__init__()
        layer = FilterLayer(args, seq_len)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)  # (256, 50, 64)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class CaserPP(nn.Module):
    def __init__(self, args, item_num, user_num, seq_len):
        super(CaserPP, self).__init__()

        # init args
        self.args = args
        self.emb_size = args.emb_size
        self.item_num = int(item_num)
        self.user_num = int(user_num)
        self.filter_sizes = list(map(int, args.filter_sizes.strip('[').strip(']').split(',')))
        self.nh = args.nh
        self.nv = args.nv
        self.dropout_rate = args.dropout_rate
        self.use_cuda = args.cuda

        self.behavior_num = 2
        self.seq_len = seq_len

        self.ac_conv = F.relu
        self.ac_fc = F.relu
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.item_encoder = Encoder(args, seq_len)

        # user、item、 behavior embeddings
        self.item_embeddings = nn.Embedding(self.item_num + 1, self.emb_size, padding_idx=self.item_num)
        self.user_embeddings = nn.Embedding(self.user_num + 1, self.emb_size, padding_idx=self.user_num)
        self.behavior_embeddings = nn.Embedding(self.behavior_num + 1, self.emb_size, padding_idx=self.behavior_num)
        self.pos_embeddings = nn.Embedding(self.seq_len + 1, self.emb_size, padding_idx=0)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.nv, (self.seq_len, 1))

        # horizontal conv layer
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.nh, (i, self.emb_size)) for i in self.filter_sizes])

        # fully-connected layer
        self.fc1_dim_v = self.nv * self.emb_size
        self.fc1_dim_h = self.nh * len(self.filter_sizes)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h

        self.fc = nn.Linear(self.emb_size * 2, self.item_num)
        self.fc1 = nn.Linear(fc1_dim_in, self.emb_size)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.behavior_embeddings.weight.data.normal_(0, 1.0 / self.behavior_embeddings.embedding_dim)
        self.pos_embeddings.weight.data.normal_(0, 1.0 / self.pos_embeddings.embedding_dim)

    def forward(self, inputs):
        item_seq, user_seq, behavior_seq, pos_seq, len_seq = inputs
        item_emb = self.item_embeddings(item_seq)
        behavior_emb = self.behavior_embeddings(behavior_seq)
        pos_emb = self.pos_embeddings(pos_seq)
        user_emb = self.user_embeddings(user_seq)

        input_emb = item_emb + behavior_emb + pos_emb
        mask = (~torch.eq(item_seq, self.item_num)).unsqueeze(-1)
        input_emb *= mask

        # filter layer
        input_encoded_layers = self.item_encoder(input_emb, output_all_encoded_layers=True)
        input_emb = input_encoded_layers[-1]

        input_emb = input_emb.unsqueeze(1).to(self.device)  # [128, 1, 50, 64]
        out, out_h, out_v = None, None, None

        # horizontal conv layer
        # Create a conv + maxpool layer for each filter size
        out_hs = []

        if self.nh:
            i = 0
            # conv: Conv2d(1, 16, kernel_size=(filter_sizes[i], 64), stride=(1, 1))
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(input_emb)).squeeze(3)
                pool_out = F.max_pool1d(conv_out, self.seq_len - self.filter_sizes[i]).squeeze(2)  # [128, 16]
                out_hs.append(pool_out)
                i += 1
            out_h = torch.cat(out_hs, 1)  # [128, 16*3] ==> [128, 48]
            # out_h = out_h.view(-1, self.fc1_dim_h)

        # vertical conv layer
        if self.nv:
            # conv_v: Conv2d(1, 4, kernel_size=(50, 1), stride=(1, 1))
            out_v = self.conv_v(input_emb)  # [128, 4, 1, 64]
            out_v = out_v.view(-1, self.fc1_dim_v)  # [128, 4*64] ==> [128, 256]

        # fully-connected Layers
        out = torch.cat([out_h, out_v], 1)  # [128, 48+256] ==> [128, 304]

        # apply dropout
        out = self.dropout(out)
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)
        output = self.fc(x)

        return output

