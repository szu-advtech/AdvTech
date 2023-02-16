import math
import jittor as jt
import jittor.nn as nn



class classification(nn.Module):
    def __init__(self, config):
        super(classification, self).__init__()
        self.model = PVT(config=config)
        self.head = nn.Linear(config.C[-1], config.num_classes) if config.num_classes > 0 else nn.Identity()
        self.apply(_init_pvt_weights)

    def execute(self, x):
        x = self.model(x)
        x = self.head(x)
        return x


# 网络结构完成
# 编写网络思路：根据论文参考结构从上至下编写网络代码
# 较复杂部分：Stage部分代码的编写，cls_token的嵌入，SR模块的编写，Attention模块多头的处理，数据在各个模块之间传输时size的变化
class PVT(nn.Module):
    def __init__(self, config):
        super(PVT, self).__init__()
        # cls_token
        self.stages = nn.ModuleList()
        Hi, Wi = config.H, config.W
        for i in range(config.num_stages):
            N_H, N_W = Hi // config.P[i], Wi // config.P[i]
            stage = Stage(i, N_H, N_W, config=config)
            self.stages.append(stage)
            Hi, Wi = Hi // config.P[i], Wi // config.P[i]
        self.norm = nn.LayerNorm(config.C[-1], eps=1e-06)

    def execute(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = self.stages(x)
        x = self.norm(x)
        return x[:, 0]


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
        for layer in range(config.num_encoder_layers[i]):
            transformer_encoder = TransformerEncoder(i, config=config)
            setattr(self, f"transformer_encoder{layer}", transformer_encoder)

    def execute(self, x):
        B, C, H, W = x.shape
        H, W = H // self.patch_embedding.patch_size, W // self.patch_embedding.patch_size
        embedding = self.patch_embedding(x)
        for layer in range(self.config.num_encoder_layers[self.i]):
            transformer_encoder = getattr(self, f"transformer_encoder{layer}")
            embedding = transformer_encoder(embedding, H, W)
        if self.i != self.config.num_stages - 1:
            res = embedding.reshape(B, self.Ci, H, W)
        else:
            res = embedding
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
        if i == last_idx:
            self.cls_token = (jt.zeros(shape=(1, 1, config.C[last_idx + 1])))
            trunc_normal(self.cls_token, std=.02)  # 初始化权重
        num_patches = N_H * N_W
        # 最后一个stage的pos_embed的num_patches需要多一个
        self.pos_embed = jt.zeros((1, num_patches, ci)) if i != config.num_stages - 1 else jt.zeros(
            (1, num_patches + 1, ci))
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        trunc_normal(self.pos_embed, std=.02)  # 初始化权重

    def execute(self, x):
        B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        H, W = H // self.patch_size, W // self.patch_size
        x = self.conv(x).flatten(2)  # b,c,h,w -> b,l,w',h'-> b,l,n
        embedding = x.permute(0, 2, 1)  # b,l,n->b,n,l
        embedding = self.norm(embedding)
        if self.i == self.config.num_stages - 1:  # 最后一个stage
            cls_token = self.cls_token.expand(B, -1, -1)
            embedding = jt.concat((cls_token, embedding), dim=1)
        # 最后一个pos_embed需要增加一维,用插值法
        # pos_embed = self.pos_embed
        if self.i != 0:  # 第一次的pos_embed不变
            # 其余的做插值处理
            if self.i == self.config.num_stages - 1:
                pos_embed_ = nn.interpolate(
                    self.pos_embed[:, 1:].reshape(1, H, W, -1).permute(0, 3, 1, 2),
                    size=(H, W), mode='bilinear').reshape(1, -1, H * W).permute(0, 2, 1)
                self.pos_embed = jt.concat((self.pos_embed[:, 0:1], pos_embed_), dim=1)
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
        self.norm1 = nn.LayerNorm(ci, eps=1e-06)
        self.attention = Attention(i, num_heads=ni, config=config)
        self.norm2 = nn.LayerNorm(ci, eps=1e-06)
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
        self.q = nn.Linear(embedding_size, embedding_size, bias=config.qkv_bias)
        self.kv = nn.Linear(embedding_size, embedding_size * 2, bias=config.qkv_bias)
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
            x_ = self.sr(x_).reshape(B, L, -1).permute(0, 2, 1)  # B,L,H,W -> H*W降低R^2倍-> B,L,N' ->B,N',L
            x_ = self.norm(x_)
            # 先将embedding 进行sr操作后再进行kv ，比直接先kv成2个embedding后再sr操作更节省算力。
            # B,N',L -kv操作-> B,N',2L->B,N',2,num_heads,head_dim ->2,B,num_heads,N',head_dim ，后面要attention相乘，最后两维必须是n,l
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, L // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, L // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # attention得分矩阵公式
        attn = (q @ k.transpose(-2, -1)) * self.scale  # n,n'
        attn = attn.softmax(dim=-1)  # 归一化
        attn = self.attn_drop(attn)
        res = attn @ v  # (B,N,N')*(B,N',L)->B,N,L
        res = res.transpose(1, 2).reshape(B, N,
                                          L)  # B,num_heads,N',head_dim ->B,N',num_heads,head_dim -> B,N,L,虽然最终的size不变，但是sr操作可以看成将R个k、R个v聚合一起求了，减少计算量。
        res = self.proj(res)  # 全连接，自动根据attention结果重新分配embedding权重
        return self.proj_drop(res)


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


def erfinv(x):
    CPU_HEADER = r'''
    float erfinv(float y) {
    float x, z, num, dem; /*working variables */
    /* coefficients in rational expansion */
    float a[4] = {  float(0.886226899), float(-1.645349621),  float(0.914624893), float(-0.140543331) };
    float b[4] = { float(-2.118377725),  float(1.442710462), float(-0.329097515),  float(0.012229801) };
    float c[4] = { float(-1.970840454), float(-1.624906493),  float(3.429567803),  float(1.641345311) };
    float d[2] = {  float(3.543889200),  float(1.637067800) };
    float y_abs = std::abs(y);
    if(y_abs > 1.0) return std::numeric_limits<float>::quiet_NaN();
    if(y_abs == 1.0) return std::copysign(std::numeric_limits<float>::infinity(), y);
    if(y_abs <= static_cast<float>(0.7)) {
    z = y * y;
    num = (((a[3]*z + a[2])*z + a[1])*z + a[0]);
    dem = ((((b[3]*z + b[2])*z + b[1])*z +b[0]) * z + static_cast<float>(1.0));
    x = y * num / dem;
    }
    else{
    z = std::sqrt(-std::log((static_cast<float>(1.0)-y_abs)/static_cast<float>(2.0)));
    num = ((c[3]*z + c[2])*z + c[1]) * z + c[0];
    dem = (d[1]*z + d[0])*z + static_cast<float>(1.0);
    x = std::copysign(num, y) / dem;
    }
    /* Two steps of Newton-Raphson correction */
    x = x - (std::erf(x) - y) / ((static_cast<float>(2.0)/static_cast<float>(std::sqrt(M_PI)))*std::exp(-x*x));
    x = x - (std::erf(x) - y) / ((static_cast<float>(2.0)/static_cast<float>(std::sqrt(M_PI)))*std::exp(-x*x));
    return(x);
    }
    '''
    return jt.code(x.shape, x.dtype, [x],
                   cpu_header=CPU_HEADER,
                   cpu_src=r'''for(int i=0;i<in0->num;i++){
                                 out0_p[i]=erfinv(in0_p[i]); 
                              }'''
                   )


def trunc_normal(tensor, mean=0., std=1., a=-2., b=2.):
    _sqrt2 = 1.4142135623730951

    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / _sqrt2)) / 2.

    with jt.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        nn.init.uniform_(tensor, 2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor = erfinv(tensor)

        # Transform to proper mean, std
        tensor = tensor * std * _sqrt2 + mean

        # Clamp to ensure it's in the proper range
        tensor = jt.clamp(tensor, min_v=a, max_v=b)
        return tensor


# ok


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

# from config import CONFIGS
# my_config = CONFIGS['PVT_classification_tiny']
# if __name__ == '__main__':
#     x = jt.randn(1, 3, 224, 224)
#     model = classification(my_config)
#     print(model)
#     # res = pvt(x)
#     # res
