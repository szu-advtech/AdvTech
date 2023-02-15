import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv, GINConv, RGCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch.nn import Parameter as Param
import numpy as np


class DiagLayer(torch.nn.Module):
    def __init__(self, in_dim, num_et=1):
        super(DiagLayer, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Param(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, x):
        # print(self.weight)
        value = x * self.weight
        return value

    def reset_parameters(self):
        self.weight.data.normal_(std=1/np.sqrt(self.in_dim))
        # self.weight.data.fill_(1)


# RGCN模型
class RGCN(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=64, output_dim_e=64, dropout=0.2, heads=10):
        super(RGCN, self).__init__()

        # 药物特征提取（3层RGCN层）
        self.rgcn1 = RGCNConv(input_dim, 128, num_relations=5, num_bases=4, aggr='mean')
        self.rgcn2 = RGCNConv(128, 64, num_relations=5, num_bases=4, aggr='mean')
        self.rgcn3 = RGCNConv(64, output_dim, num_relations=5, num_bases=4, aggr='mean')
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # 药物副作用特征提取（使用3层GAT层）
        self.gat1 = GATConv(input_dim_e, 128, heads=heads, dropout=dropout)
        self.gat2 = GATConv(128 * heads, output_dim, heads=heads, dropout=dropout)
        self.gat3 = GATConv(output_dim * heads, output_dim, dropout=dropout)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # 激活函数
        self.relu = nn.ReLU()
        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, data, data_e, DF=False, not_FC=True):
        # graph input feed-forward
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index
        # print(x.shape)
        # 药物
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.tanh(self.rgcn1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.tanh(self.rgcn2(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.rgcn3(x, edge_index, edge_type)
        x = torch.tanh(x)
        # 全局最大池化
        x = global_max_pool(x, batch)

        # 副作用
        x_e = self.gat1(x_e, edge_index_e)
        x_e = self.relu(x_e)
        x_e = self.gat2(x_e, edge_index_e)
        x_e = self.relu(x_e)
        x_e = self.gat3(x_e, edge_index_e)
        x_e = self.relu(x_e)

        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x

        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e

# GAT三层模型
class GAT3(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, output_dim_e=64, dropout=0.2, heads=10):
        super(GAT3, self).__init__()

        # graph layers : drug
        self.gcn1 = GATConv(input_dim, 128, heads=heads, dropout=dropout)
        self.gcn2 = GATConv(128 * heads, output_dim, heads=heads, dropout=dropout)
        self.gcn5 = GATConv(output_dim * heads, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # # graph layers : sideEffect
        self.gcn3 = GATConv(input_dim_e, 128, heads=heads, dropout=dropout)
        self.gcn4 = GATConv(128 * heads, output_dim, heads=heads, dropout=dropout)
        self.gcn6 = GATConv(output_dim * heads, output_dim, dropout=dropout)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()
        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, data, data_e, DF=False, not_FC=True):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index
        # 药物
        x = self.relu(self.gcn1(x, edge_index))
        x = self.relu(self.gcn2(x, edge_index))
        x = self.relu(self.gcn5(x, edge_index))
        x = global_max_pool(x, batch)  # global max pooling

        # 副作用
        x_e = self.relu(self.gcn3(x_e, edge_index_e))
        x_e = self.relu(self.gcn4(x_e, edge_index_e))
        x_e = self.relu(self.gcn6(x_e, edge_index_e))

        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x

        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e
