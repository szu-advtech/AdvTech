import torch
import numpy as np
from torch import nn
import math
from ECB import ECB

# ############################### edgeSR ###############################
class edgeSR_MAX(nn.Module):
    def __init__(self,  scale_factor, k=7, num_channels=9, groups=1):
        super().__init__()
        self.channels = num_channels
        self.kernel_size = (k, k)
        self.stride = scale_factor

        self.pixel_shuffle = nn.PixelShuffle(self.stride)
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=(self.stride**2)*self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(
                (self.kernel_size[0]-1)//2,
                (self.kernel_size[1]-1)//2
            ),
            groups=groups,
            bias=False,
            dilation=1
        )
        nn.init.xavier_normal_(self.filter.weight, gain=1.)
        self.filter.weight.data[:, 0, self.kernel_size[0]//2, self.kernel_size[0]//2] = 1.

    def forward(self, input):
        return self.pixel_shuffle(self.filter(input)).max(dim=1, keepdim=True)[0]



class edgeSR_TM(nn.Module):
    def __init__(self, scale_factor, k=7, num_channels=9, groups=1):
        super().__init__()
        self.channels = num_channels
        self.kernel_size = (k, k)
        self.stride = scale_factor

        self.pixel_shuffle = nn.PixelShuffle(self.stride)
        self.softmax = nn.Softmax(dim=1)
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=2*(self.stride**2)*self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(
                (self.kernel_size[0]-1)//2,
                (self.kernel_size[1]-1)//2
            ),
            groups=groups,
            bias=False,
            dilation=1
        )
        nn.init.xavier_normal_(self.filter.weight, gain=1.)
        self.filter.weight.data[:, 0, self.kernel_size[0]//2, self.kernel_size[0]//2] = 1.

    def forward(self, input):
        filtered = self.pixel_shuffle(self.filter(input))
        value, key = torch.split(filtered, [self.channels, self.channels], dim=1)
        return torch.sum(
            value * self.softmax(key),
            dim=1, keepdim=True
        )


class edgeSR_TR(nn.Module):
    def __init__(self, scale_factor, k=7, num_channels=9, groups=1):
        super().__init__()

        self.channels = num_channels
        self.kernel_size = (k, k)
        self.stride = scale_factor

        self.pixel_shuffle = nn.PixelShuffle(self.stride)
        self.softmax = nn.Softmax(dim=1)
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=3*(self.stride**2)*self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(
                (self.kernel_size[0]-1)//2,
                (self.kernel_size[1]-1)//2
            ),
            groups=groups,
            bias=False,
            dilation=1
        )
        nn.init.xavier_normal_(self.filter.weight, gain=1.)
        self.filter.weight.data[:, 0, self.kernel_size[0]//2, self.kernel_size[0]//2] = 1.

    def forward(self, input):
        filtered = self.pixel_shuffle(self.filter(input))
        value, query, key = torch.split(filtered, [self.channels, self.channels, self.channels], dim=1)
        return torch.sum(
            value * self.softmax(query*key),
            dim=1, keepdim=True
        )


class edgeSR_TR_ECBSR(nn.Module):
    def __init__(self, scale_factor, k=7, num_channels=7, groups=1, flag = 1):
        super().__init__()

        self.channels = num_channels
        # self.kernel_size = (k, k)
        self.stride = scale_factor

        # self.pixel_shuffle = nn.PixelShuffle(self.stride)
        self.softmax = nn.Softmax(dim=1)
        # module_nums, channel_nums, with_idt, act_type, scale, colors
        self.ecbsr = ECBSR(4, 16, False, 'prelu', self.stride, self.channels, flag=flag)

        # self.filter = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=3*(self.stride**2)*self.channels,
        #     kernel_size=self.kernel_size,
        #     stride=1,
        #     padding=(
        #         (self.kernel_size[0]-1)//2,
        #         (self.kernel_size[1]-1)//2
        #     ),
        #     groups=groups,
        #     bias=False,
        #     dilation=1
        # )
        # nn.init.xavier_normal_(self.filter.weight, gain=1.)
        # self.filter.weight.data[:, 0, self.kernel_size[0]//2, self.kernel_size[0]//2] = 1.

    def forward(self, input):
        # filtered = self.pixel_shuffle(self.ecbsr(input))
        filtered = self.ecbsr(input)
        value, query, key = torch.split(filtered, [self.channels, self.channels, self.channels], dim=1)
        return torch.sum(
            value * self.softmax(query*key),
            dim=1, keepdim=True
        )



class edgeSR_CNN(nn.Module):
    def __init__(self, scale_factor, num_channels=7, d=9, s=5, groups=1):
        super().__init__()
        self.channels = num_channels
        self.stride = scale_factor
        D = d
        S = s
        assert S > 0 and D >= 0
        self.softmax = nn.Softmax(dim=1)
        if D == 0:
            self.filter = nn.Sequential(
                nn.Conv2d(D, S, (3, 3), (1, 1), (1, 1)),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=S,
                    out_channels=2 * (self.stride ** 2) * self.channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    groups=groups,
                    bias=False,
                    dilation=1
                ),
                nn.PixelShuffle(self.stride),
            )
        else:
            self.filter = nn.Sequential(
                nn.Conv2d(1, D, (5, 5), (1, 1), (2, 2)),
                nn.Tanh(),
                nn.Conv2d(D, S, (3, 3), (1, 1), (1, 1)),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=S,
                    out_channels=2*(self.stride ** 2)*self.channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    groups=groups,
                    bias=False,
                    dilation=1
                ),
                nn.PixelShuffle(self.stride),
            )

        if D == 0:
            nn.init.xavier_normal_(self.filter[0].weight, gain=1.)
            nn.init.xavier_normal_(self.filter[2].weight, gain=1.)
            self.filter[0].weight.data[:, 0, 1, 1] = 1.
            self.filter[2].weight.data[:, 0, 1, 1] = 1.
        else:
            nn.init.xavier_normal_(self.filter[0].weight, gain=1.)
            nn.init.xavier_normal_(self.filter[2].weight, gain=1.)
            nn.init.xavier_normal_(self.filter[4].weight, gain=1.)
            self.filter[0].weight.data[:, 0, 2, 2] = 1.
            self.filter[2].weight.data[:, 0, 1, 1] = 1.
            self.filter[4].weight.data[:, 0, 1, 1] = 1.

    def forward(self, input):
        filtered = self.filter(input)
        value, key = torch.split(filtered, [self.channels, self.channels], dim=1)
        return torch.sum(
            value * self.softmax(key*key),
            dim=1, keepdim=True
        )



# ############################### ESPCN ###############################
class ESPCN(nn.Module):
    def __init__(self, scale_factor, d=64, s=32):
        super().__init__()
        self.stride = scale_factor
        D = d
        S = s
        if D == 0:
            self.net = nn.Sequential(
                nn.Conv2d(1, S, (3, 3), (1, 1), (1, 1)),
                nn.Tanh(),
                nn.Conv2d(S, self.stride**2, (3, 3), (1, 1), (1, 1)),
                nn.PixelShuffle(self.stride),
                nn.Sigmoid(),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(1, D, (5, 5), (1, 1), (2, 2)),
                nn.Tanh(),
                nn.Conv2d(D, S, (3, 3), (1, 1), (1, 1)),
                nn.Tanh(),
                nn.Conv2d(S, self.stride**2, (3, 3), (1, 1), (1, 1)),
                nn.PixelShuffle(self.stride),
                nn.Sigmoid(),
            )

    def forward(self, input):
        return self.net(input)

# ############################### FSRCNN ###############################

class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super().__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(1, d, 5, padding=2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)


    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


# ############################### ECBSR ###############################
class ECBSR(nn.Module):
    def __init__(self, module_nums, ECB_channel_nums, with_idt, act_type, scale, eSR_TR_channels, flag):
        super(ECBSR, self).__init__()
        self.module_nums = module_nums
        self.ECB_channel_nums = ECB_channel_nums
        self.scale = scale
        self.eSR_TR_channels = eSR_TR_channels
        self.with_idt = with_idt
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [ECB(1, self.ECB_channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, flag=flag)]
        for i in range(self.module_nums):
            backbone += [ECB(self.ECB_channel_nums, self.ECB_channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, flag=flag)]
        backbone += [ECB(self.ECB_channel_nums, 3 * self.eSR_TR_channels*self.scale*self.scale, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt, flag=flag)]

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)

    def forward(self, x,):
        y = self.backbone(x) + x
        y = self.upsampler(y)
        return y














# ######################  序列推荐  ##############################
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)








# ############################### 其他功能函数 ###############################





















