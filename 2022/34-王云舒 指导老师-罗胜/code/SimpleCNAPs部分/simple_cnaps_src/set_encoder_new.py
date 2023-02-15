import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

"""
    Classes and functions required for Set encoding in adaptation networks. Many of the ideas and classes here are
    closely related to DeepSets (https://arxiv.org/abs/1703.06114).
    自适应网络中Set编码所需的类和函数。这里的许多想法和类都与DeepSets (https://arxiv.org/abs/1703.06114)密切相关。
"""


def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)


# def regu(x, way, shot):
#     idx = torch.Tensor(np.arange(way*shot)).long().view(1, shot, way)
#     temp_x = x[idx.contiguous().view(-1)].contiguous().view(*(idx.shape + (-1,)))
#     proto = temp_x.mean(dim=1)
#     return proto

class SetEncoder(nn.Module):
    """
    Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
    on sets (mainly for extracting task-level representations from context sets).
    简单的集编码器，实现DeepSets方法。用于对集合上的置换不变表示进行建模(主要用于从上下文集合中提取任务级表示)。
    """

    def __init__(self):
        super(SetEncoder, self).__init__()
        self.pre_pooling_fn = SimplePrePoolNet()
        # self.pre_pooling_fn = ResidualPrePoolNet()
        # self.pre_pooling_fn = ConvNet()
        # self.pre_pooling_fn = MultiHeadAttention(8, 3*224*224, 64, 64, dropout=0.5)
        # self.se = SELayer(64)
        # self.attn_fn = MultiHeadAttention(8, 64, 64, 64, dropout=0.5)
        self.pooling_fn = mean_pooling
        self.post_pooling_fn = Identity()
        # self.regu = regu

    def forward(self, x):
        """
        Forward pass through DeepSet SetEncoder. Implements the following computation: 前向通过DeepSet SetEncoder。实现以下计算
        g(X) = rho ( mean ( phi(x) ) )
        Where X = (x0, ... xN) is a set of elements x in X (in our case, images from a context set)
        and the mean is a pooling operation over elements in the set.
        其中X = (x0，…xN)是x中的x元素的集合(在我们的例子中，是来自上下文集合的图像)，而均值是集合中元素的池化操作。
        :param x: (torch.tensor) Set of elements X (e.g., for images has shape batch x C x H x W ).
        :return: (torch.tensor) Representation of the set, single vector in Rk. 集合的表示，Rk中的单个向量。
        """
        x = self.pre_pooling_fn(x)
        # x = self.se(x)
        # x = self.regu(x, way, shot)
        # x = self.attn_fn(x,x,x)
        x = self.pooling_fn(x)
        x = self.post_pooling_fn(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out += identity
        out = self.relu(out)
        out = self.pool(out)

        return out


class ResidualPrePoolNet(nn.Module):
    """
    Simple prepooling network for images. Implements the phi mapping in DeepSets networks. In this work we use a
    multi-layer convolutional network similar to that in https://openreview.net/pdf?id=rJY0-Kcll.
    """

    def __init__(self):
        super(ResidualPrePoolNet, self).__init__()
        self.layer1 = self._make_conv2d_layer(3, 64)
        self.layer2 = BasicBlock(64, 64)
        self.layer3 = BasicBlock(64, 64)
        self.layer4 = BasicBlock(64, 64)
        self.layer5 = BasicBlock(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps):
        return nn.Sequential(
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_maps),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    @property
    def output_size(self):
        return 64


class SimplePrePoolNet(nn.Module):
    """
    Simple prepooling network for images. Implements the phi mapping in DeepSets networks. In this work we use a
    multi-layer convolutional network similar to that in https://openreview.net/pdf?id=rJY0-Kcll.
    简单的图像预池网络。在DeepSets网络中实现phi映射。在本工作中，我们使用了一个类似于https://openreview.net/pdf?id=rJY0-Kcll中的多层卷积网络。
    """

    def __init__(self):
        super(SimplePrePoolNet, self).__init__()
        self.layer1 = self._make_conv2d_layer(3, 64)
        self.layer2 = self._make_conv2d_layer(64, 64)
        self.layer3 = self._make_conv2d_layer(64, 64)
        self.layer4 = self._make_conv2d_layer(64, 64)
        self.layer5 = self._make_conv2d_layer(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.residual = self.residual_block()

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps):
        return nn.Sequential(
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_maps),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    @property
    def output_size(self):
        return 64


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        # y = self.avg_pool(x).view(b, c)
        y = self.fc(x).view(b, c)
        return x * y.expand_as(x)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        len_q, _ = q.size()
        len_k, _ = k.size()
        len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(len_q, n_head, d_k)
        k = self.w_ks(k).view(len_k, n_head, d_k)
        v = self.w_vs(v).view(len_v, n_head, d_v)

        q = q.permute(1, 0, 2).contiguous()  # (n*b) x lq x dk
        k = k.permute(1, 0, 2).contiguous()  # (n*b) x lk x dk
        v = v.permute(1, 0, 2).contiguous()  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        # output = output.view(n_head, len_q, d_v)
        output = output.permute(1, 0, 2).contiguous().view(len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        # output = output.view(len_q,_)

        return output

    @property
    def output_size(self):
        return 64


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = nn.MaxPool2d(5)(x)
        x = x.view(x.size(0), -1)
        return x

    @property
    def output_size(self):
        return 64
