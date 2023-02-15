import jittor as jt
import jittor.nn as nn
import jittor.transform.function_pil as F
from jittor.init import trunc_normal_


# from config import CONFIGS
# config = CONFIGS['PVT']

class segmentation(nn.Module):
    def __init__(self, config):
        super(segmentation, self).__init__()
        self.pvt = PVT(config=config)

    def execute(self, x):
        x = self.pvt(x)
        return x


# 网络结构完成
# 编写网络思路：根据论文参考结构从上至下编写网络代码
# 较复杂部分：Stage部分代码的编写，SR模块的编写，Attention模块多头的处理，数据在各个模块之间传输时size的变化
class PVT(nn.Module):
    def __init__(self, config):
        super(PVT, self).__init__()
        self.stages = nn.ModuleList()
        self.F4 = config.F4
        Hi, Wi = config.H, config.W
        for i in range(config.num_stages):
            N_H, N_W = Hi // config.P[i], Wi // config.P[i]
            stage = Stage(i, N_H, N_W, config=config)
            self.stages.append(stage)
            Hi, Wi = Hi // config.P[i], Wi // config.P[i]
        self.norm = nn.LayerNorm(config.C[-1])
        # self.apply(_init_pvt_weights)

    def execute(self, x):
        B, H, W, C = x.shape
        # x = x.permute(0, 3, 1, 2)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        if self.F4:
            x = outs[3:4]
            return x
        return outs


class Stage(nn.Module):
    def __init__(self, i, N_H, N_W, config):
        super(Stage, self).__init__()
        self.config = config
        self.i = i
        self.N_H, self.N_W = N_H, N_W
        self.Ci = config.C[i + 1]
        self.Pi = config.P[i]
        # Patch_Embedding
        self.patch_embedding = Patch_Embedding(i, N_H, N_W, config=config)

        # TransformerEncoder layers
        # self.layers = nn.Sequential()
        for layer in range(config.num_encoder_layers[i]):
            transformer_encoder = TransformerEncoder(i, config=config)
            setattr(self, f"transformer_encoder{layer}", transformer_encoder)

    def execute(self, x):
        B, C, H, W = x.shape
        embedding = self.patch_embedding(x)
        H, W = H // self.patch_embedding.patch_size, W // self.patch_embedding.patch_size
        for layer in range(self.config.num_encoder_layers[self.i]):
            transformer_encoder = getattr(self, f"transformer_encoder{layer}")
            embedding = transformer_encoder(embedding, H, W)
        # embedding = self.layers(embedding, self.N_H, self.N_W)
        res = embedding.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return res


# ok
class Patch_Embedding(nn.Module):
    def __init__(self, i, N_H, N_W, config):
        super(Patch_Embedding, self).__init__()
        self.config = config
        self.patch_size = config.P[i]
        ci_1 = config.C[i]
        ci = config.C[i + 1]
        self.i = i
        # Patch_Embedding
        # Linear+Reshape
        self.conv = nn.Conv2d(in_channels=ci_1, out_channels=ci, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(ci)

        last_idx = config.num_stages - 1

        num_patches = N_H * N_W
        # 最后一个stage的pos_embed的num_patches需要多一个
        self.pos_embed = nn.Parameter(jt.zeros((1, num_patches, ci))) if i != config.num_stages - 1 else nn.Parameter(
            jt.zeros((1, num_patches + 1, ci)))
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        trunc_normal_(self.pos_embed, std=.02)  # 初始化权重

    def execute(self, x):
        B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        H, W = H // self.patch_size, W // self.patch_size
        x = self.conv(x).flatten(2)  # b,c,h,w -> b,l,w',h'-> b,l,n
        embedding = x.permute(0, 2, 1)  # b,l,n->b,n,l
        embedding = self.norm(embedding)

        if self.i == self.config.num_stages - 1:  # 最后一个stage
            self.pos_embed = nn.interpolate(
                self.pos_embed[:, 1:].reshape(1, H, W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode='bilinear').reshape(1, -1, H * W).permute(0, 2, 1)
        else:
            self.pos_embed = nn.interpolate(
                self.pos_embed.reshape(1, H, W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode='bilinear').reshape(1, -1, H * W).permute(0, 2, 1)
        embedding = self.pos_drop(embedding + self.pos_embed)
        return embedding


# ok
class TransformerEncoder(nn.Module):
    def __init__(self, i, config):
        super(TransformerEncoder, self).__init__()
        ci = config.C[i + 1]
        ni = config.num_heads[i]
        self.norm1 = nn.LayerNorm(ci)
        self.attention = Attention(i, num_heads=ni, config=config)
        self.norm2 = nn.LayerNorm(ci)
        mlp_hidden_dim = int(config.mlp_rate[i] * ci)
        self.mlp = MLP(ci, mlp_hidden_dim, ci, config)

    def execute(self, x, H, W):
        norm1 = self.norm1(x)
        attn = self.attention(norm1, H, W)
        norm2 = self.norm2(x + attn)
        res = attn + self.mlp(norm2)
        return res


# ok
class Attention(nn.Module):
    def __init__(self, i, num_heads, config):
        super(Attention, self).__init__()
        embedding_size = config.C[i + 1]
        assert embedding_size % num_heads == 0, f"embedding长度 {embedding_size} 必须可以被多头数 {num_heads} 整除。"
        self.num_heads = config.num_heads[i]
        head_dim = embedding_size // num_heads
        self.scale = config.qk_scale or head_dim ** -0.5
        self.q = nn.Linear(embedding_size, embedding_size, bias=config.qk_bias)
        self.kv = nn.Linear(embedding_size, embedding_size * 2, bias=config.qk_bias)
        self.attn_drop = nn.Dropout(config.drop_rate)
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.proj_drop = nn.Dropout(config.drop_rate)
        self.ri = config.R[i]
        if self.ri > 1:
            self.sr = nn.Conv2d(in_channels=embedding_size, out_channels=embedding_size, kernel_size=self.ri,
                                stride=self.ri)
            self.norm = nn.LayerNorm(embedding_size)

    def execute(self, x, H, W):
        # x = x.permute(0, 2, 1)  # b,n,l->b,l,n
        B, N, L = x.shape  # L=H*W
        # 为了多头注意力可以使用，最后需要与shape为
        q = self.q(x).reshape(B, N, self.num_heads, L // self.num_heads).permute(0, 2, 1, 3)  # B,num_heads,N,head_dim

        # sr模块
        # 先从一维变回二维，才能送进sr去卷积，这样H*W总共降低了R^2倍
        if self.ri > 1:
            x_ = x.permute(0, 2, 1).reshape(B, L, H, W)  # B,L,N->B,L,H,W
            x_ = self.sr(x_).reshape(B, L, -1).permute(0, 2, 1)  # B,L,H,W -H*W降低R^2倍-> B,L,N' ->B,N',L
            x_ = self.norm(x_)
            # 先将embedding 进行sr操作后再进行kv ，比直接先kv成2个embedding后再sr操作更节省算力。
            # B,N',L -kv操作-> B,N',2L->B,N',2,num_heads,head_dim ->2,B,num_heads,N',head_dim ，后面要attention相乘，最后两维必须是n,l
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, L // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, L // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # attention得分矩阵公式
        attn = (q @ k.transpose(-2, -1)) * self.scale  # n,n
        attn = attn.softmax(dim=-1)  # 归一化
        attn = self.attn_drop(attn)
        res = attn @ v  # B,num_heads,N',head_dim
        res = res.transpose(1, 2).reshape(B, N,
                                          L)  # B,num_heads,N',head_dim ->B,N',num_heads,head_dim -> B,N,L,虽然最终的size不变，但是sr操作可以看成将R个k、R个v聚合一起求了，减少计算量。
        res = self.proj(res)  # 全连接，自动根据attention结果重新分配embedding权重
        return self.proj_drop(res)


# ok
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, config, act_fn=nn.GELU):
        super(MLP, self).__init__()
        self.process = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            act_fn(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.Dropout(config.drop_rate)
        )

    def execute(self, x) -> None:
        x = self.process(x)
        return x


def _init_pvt_weights(m):
    """
    PVT weight initialization
    :param m: module

    """
    # if isinstance(m, nn.Linear):
    #     trunc_normal_(m.weight, std=.02)
    #     if isinstance(m, nn.Linear) and m.bias is not None:
    #         nn.init.constant_(m.bias, 0)
    # elif isinstance(m, nn.LayerNorm):
    #     nn.init.constant_(m.bias, 0)
    #     nn.init.constant_(m.weight, 1.0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zero_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zero_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zero_(m.bias)
        nn.init.one_(m.weight)


if __name__ == '__main__':
    x = jt.randn(1, 3, 224, 224)
    pvt = PVT(H=x.shape[2], W=x.shape[3])
    res = pvt(x)
    res
