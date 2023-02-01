import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
import numpy

'''
if you have any questions
you can sent an e-mail to me
akieqi@qq.com
'''

class Mycon(MessagePassing):
    def __init__(self):
        super(Mycon, self).__init__(aggr='mean')
    # def __len__(self):
    #     super(Mycon, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def update(self, inputs: Tensor) -> Tensor:
        return inputs

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


class MGNN(nn.Module):
    def __init__(self, graph, hidden_num, node_num):
        super(MGNN, self).__init__()
        torch.manual_seed(123456)
        self.con = Mycon()
        self.node_num = torch.tensor(node_num)

        self.graph_0 = nn.Parameter(torch.LongTensor(graph[0]), requires_grad=False)
        self.graph_1 = nn.Parameter(torch.LongTensor(graph[1]), requires_grad=False)
        self.graph_2 = nn.Parameter(torch.LongTensor(graph[2]), requires_grad=False)
        self.graph_3 = nn.Parameter(torch.LongTensor(graph[3]), requires_grad=False)

        self.hidden_num = torch.tensor(hidden_num)

        self.node_embedding = nn.Parameter(torch.rand([self.node_num, self.hidden_num])/100)
        self.Wg = nn.Parameter(torch.rand([1, 2 * self.hidden_num])/1000)
        self.W = nn.Parameter(torch.rand([self.hidden_num, self.hidden_num]))     # (8)


        self.node_h1_emb = self.node_embedding + self.Conv(self.graph_0, self.graph_1, self.graph_2, self.graph_3, self.node_embedding)
        self.node_h2_emb = self.node_h1_emb + self.Conv(self.graph_0, self.graph_1, self.graph_2, self.graph_3, self.node_h1_emb)


    def update_h1h2(self):
        self.node_h1_emb = self.node_embedding + self.Conv(self.graph_0, self.graph_1, self.graph_2, self.graph_3, self.node_embedding)
        self.node_h2_emb = self.node_h1_emb + self.Conv(self.graph_0, self.graph_1, self.graph_2, self.graph_3, self.node_h1_emb)

    def Conv(self, graph_0_in, graph_0_out, graph_1_in, graph_1_out, node_emb):
        # h = node_emb
        h_0 = self.con(node_emb, graph_0_in)
        h_1 = self.con(node_emb, graph_0_out)
        h_2 = self.con(node_emb, graph_1_in)
        h_3 = self.con(node_emb, graph_1_out)
        h = h_0 + h_1 + h_2 + h_3
        return h




    def forward(self, buy_list, click_list, target, neg):

        node_idx = torch.cat((neg, target), dim=1).long()
        # click_emb = self.node_h2_emb[click_list].mean(axis=1)
        # buy_emb = self.node_h2_emb[buy_list].mean(axis=1)
        click_emb = self.Embedding(click_list)
        buy_emb = self.Embedding(buy_list)
        node_emb = self.node_embedding[node_idx].permute((0, 2, 1))

        alpha = torch.sigmoid(torch.mm(self.Wg, torch.cat((click_emb, buy_emb), dim=1).T))[0]
        o = torch.stack([alpha[i] * click_emb[i] + (1-alpha[i]) * buy_emb[i] for i in range(len(alpha))])
        s = torch.bmm(torch.unsqueeze(torch.mm(o, self.W), 0).permute((1, 0, 2)),node_emb)
        y = s.permute((1, 0, 2))[0]
        label = torch.tensor(numpy.append([0 for i in neg[0]], 1)).expand((len(neg), len(neg[0]) + 1)).cuda()
        return y, label

    def Embedding(self, seq):
        '''
        返回去掉0标记的最终序列embedding

        :param seq: 序列 [batch_size,L]
        :return: h2的embedding [batch_size,dim_k]
        '''
        mark = torch.ne(seq, 0).float()
        h2_emb = self.node_h2_emb[seq]
        emb = (h2_emb * torch.unsqueeze(mark, 2)).sum(axis=1)
        total = torch.unsqueeze(mark.sum(axis=1),1)
        return emb/total


    def topk(self,buy_list,click_list):
        # click_list = x[0]  # list
        # buy_list = x[1]
        click_emb = self.Embedding(click_list)
        buy_emb = self.Embedding(buy_list)
        # click_emb = self.node_h2_emb[click_list].mean(axis=1)
        # buy_emb = self.node_h2_emb[buy_list].mean(axis=1)

        alpha = torch.sigmoid(torch.mm(self.Wg,torch.cat((click_emb,buy_emb), dim=1).T))[0]
        o = torch.stack([alpha[i] * click_emb[i] + (1-alpha[i]) * buy_emb[i] for i in range(len(alpha))])
        s = torch.mm(torch.mm(o, self.W), self.node_embedding.T)
        y = s
        # y = s.permute((1, 0, 2))[0]

        topk = torch.topk(y, k=500)
        topkV = topk.values
        topkI = topk.indices

        return topkV, topkI
