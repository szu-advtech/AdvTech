from core.model.net_utils import FC, MLP, LayerNorm
import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C
        # 定义三个线性层，分别对 value，key 和 query 做线性变换
        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        # 定义 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        # 对 value 做线性变换，并 reshape 为四维张量
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)
        # 对 key 做线性变换，并 reshape 为四维张量
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)
        # 对 query 做线性变换，并 reshape 为四维张量
        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)
        # 计算多头注意力
        atted = self.att(v, k, q, mask)
        # 重新 reshape 为三维张量
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        # 对多头注意力结果做线性变换
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        # 计算每个询问的维度
        d_k = query.size(-1)

        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 如果有mask，使用mask进行填充
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        # 计算softmax并进行dropout
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        # 计算加权平均值
        return torch.matmul(att_map, value)

# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        # 实例化一个多层感知机
        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,  # 输入维度
            mid_size=__C.FF_SIZE,  # 中间隐藏层维度
            out_size=__C.HIDDEN_SIZE,  # 输出维度
            dropout_r=__C.DROPOUT_R,  # dropout比率
            use_relu=True  # 使用ReLU激活函数
        )

    def forward(self, x):
        return self.mlp(x)

# ------------------------
# ---- Self Attention ----
# ------------------------
# 自注意力
class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        # 实例化一个多头注意力层
        self.mhatt = MHAtt(__C)
        # 实例化一个FFN层
        self.ffn = FFN(__C)

        # 定义两个dropout层和两个归一化层
        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        # 计算多头注意力输出
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))
        # 计算FFN输出
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x

# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        # 实例化两个多头注意力模块
        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)

        # 实例化全连接层
        self.ffn = FFN(__C)

        # 实例化Dropout层和LayerNorm层，用于数据预处理
        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        # 第一次计算多头注意力
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        # 第二次计算多头注意力
        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        # 全连接层计算
        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        # 创建多层Encoder和Decoder
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # 经过多层Encoder计算隐藏向量
        for enc in self.enc_list:
            x = enc(x, x_mask)

        # 经过多层Decoder计算答案
        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y
