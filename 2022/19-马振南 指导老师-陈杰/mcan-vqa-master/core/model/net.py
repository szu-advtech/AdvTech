from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------
class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        # 定义一个MLP模型
        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,  # 输入的维数为__C.HIDDEN_SIZE
            mid_size=__C.FLAT_MLP_SIZE,  # 中间层维数为__C.FLAT_MLP_SIZE
            out_size=__C.FLAT_GLIMPSES,  # 输出维数为__C.FLAT_GLIMPSES
            dropout_r=__C.DROPOUT_R,  # dropout比例为__C.DROPOUT_R
            use_relu=True  # 使用ReLU激活函数
        )

        # 定义一个线性层，用于合并所有的注意力特征
        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,  # 输入维数为__C.HIDDEN_SIZE * __C.FLAT_GLIMPSES
            __C.FLAT_OUT_SIZE  # 输出维数为__C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        # 计算注意力权重
        att = self.mlp(x)
        # 将mask为0的位置的注意力权重设为负无穷大
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        # 对注意力权重进行softmax
        att = F.softmax(att, dim=1)

        att_list = []
        # 将每一个注意力特征加权求和
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        # 将所有的注意力特征拼接起来
        x_atted = torch.cat(att_list, dim=1)
        # 对所有的注意力特征进行线性变换
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        # 初始化词嵌入层，其中 token_size 表示词典中的单词数量，__C.WORD_EMBED_SIZE 表示嵌入向量的维数
        self.embedding = nn.Embedding(num_embeddings=token_size, embedding_dim=__C.WORD_EMBED_SIZE)

        # 加载 GloVe 词向量权重
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # 初始化 LSTM 层
        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # 初始化将图像特征从 IMG_FEAT_SIZE 维映射到 HIDDEN_SIZE 维的线性层
        self.img_feat_linear = nn.Linear(__C.IMG_FEAT_SIZE, __C.HIDDEN_SIZE)

        # 初始化 MCA_ED 层
        self.backbone = MCA_ED(__C)

        # 初始化两个 AttFlat 层，分别对应图像特征和语言特征
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        # 初始化 LayerNorm 层和线性层
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def forward(self, img_feat, ques_ix):
        # 制作语言特征和图像特征的掩码
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # 预处理语言特征
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # 预处理图像特征
        img_feat = self.img_feat_linear(img_feat)

        # MCA_ED模型
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # 对语言特征进行注意力映射
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        # 对图像特征进行注意力映射
        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # 计算最终特征
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat

    #掩码生成函数
    def make_mask(self, feature):
        # 计算特征的绝对值的和
        # 如果该特征的绝对值的和为0，则认为该特征为padding部分
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

