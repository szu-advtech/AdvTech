import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph  # 建模由open pose提取出来的骨架数据


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        """

        load graph ,graph包括:
        layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)  # 获得edge和center
        self.hop_dis = get_hop_distance(  # Numpy数组，邻接矩阵，到自身是0，能到是1，不能到是inf
            self.num_node, self.edge, max_hop=max_hop)  # num_node=25 edge = [(0,0),...,(24,24),(0,1),...,(23,11)]
        self.get_adjacency(strategy)  # 一个三维的标准化邻接矩阵

        """
        self.graph = Graph(**graph_args)  # graph_args:{layout: 'ntu-rgb+d, 'strategy: 'spatial'}

        """
        requires_grad: 如果需要为张量计算梯度，则为True，否则为False
        """

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)  # 将numpy数组（三维邻接矩阵A）转成tensor
        self.register_buffer('A', A)  # 定义A在模型训练时不会更新，只可人为地改变它们的值，但是该组参数又作为模型参数不可或缺的一部分。

        # build networks
        spatial_kernel_size = A.size(0)  # 整型变量，表第0维元素的个数,spatial_kernel_size = 3
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)  # 元组(9, 3)
        # class torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True) [source]
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))  # in_channels = 3, A.size(1) = 25 ,25*16表示期望输入的特征数
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}  # 将定义的形参和它们对应的实参用字典型关联起来
        self.st_gcn_networks = nn.ModuleList((  # 它可以以列表的形式来保持多个子模块。
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),  # 加一层
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),  # 加一层
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        self.fc = nn.Linear(hidden_dim, num_class)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        # prediction
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`, K = 3
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,  # hidden_channels: 16
                 kernel_size,  # 元组(9, 3)
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)  # padding = (1, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])  # 3,16,3

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A