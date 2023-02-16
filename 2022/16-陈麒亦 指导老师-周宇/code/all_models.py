import torch
from torch import nn
from torch_geometric.nn import GCNConv,SAGEConv,RGATConv,RGCNConv
from torch_geometric.nn import GATConv
import torch.nn.functional as F

torch.backends.cudnn.enabled = False

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphCDA(nn.Module):
    def __init__(self,head_nums):
        super(GraphCDA,self).__init__()

        self.gcn_cir_f1 = GCNConv(128, 128)
        self.gat_cir_f1 = GATConv(128, 128, heads=head_nums, concat=False, edge_dim=1)
        self.gcn_cir_f2 = GCNConv(128, 128)

        self.gcn_dis_f1 = GCNConv(128, 128)
        self.gat_dis_f1 = GATConv(128, 128, heads=head_nums, concat=False, edge_dim=1)
        self.gcn_dis_f2 = GCNConv(128, 128)

        self.cnn_cir = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=(128, 1), stride=(1, 1), bias=True)
        self.cnn_dis = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=(128, 1), stride=(1, 1), bias=True)

    def forward(self,x):
        torch.manual_seed(1)
        x_cir=torch.randn(585,128).to(device)
        x_dis=torch.randn(88,128).to(device)

        cc_matrix=x['cc_matrix'].to(device)
        cc_edges = x['cc_edges'].to(device)
        dd_matrix = x['dd_matrix'].to(device)
        dd_edges = x['dd_edges'].to(device)

        x_cir_f1=self.gcn_cir_f1(x_cir,cc_edges,cc_matrix[cc_edges[0],cc_edges[1]])
        x_cir_f1=torch.relu(x_cir_f1)
        x_cir_att = self.gat_cir_f1(x_cir_f1, cc_edges,cc_matrix[cc_edges[0], cc_edges[1]])
        x_cir_att = torch.relu(x_cir_att)
        x_cir_f2 = self.gcn_cir_f2(x_cir_att, cc_edges,cc_matrix[cc_edges[0], cc_edges[1]])
        x_cir_f2 = torch.relu(x_cir_f2)
        #[585,128]

        x_dis_f1 = self.gcn_dis_f1(x_dis, dd_edges,dd_matrix[dd_edges[0], dd_edges[1]])
        x_dis_f1 = torch.relu(x_dis_f1)
        x_dis_att = self.gat_dis_f1(x_dis_f1, dd_edges,dd_matrix[dd_edges[0], dd_edges[1]])
        x_dis_att = torch.relu(x_dis_att)
        x_dis_f2 = self.gcn_dis_f2(x_dis_att, dd_edges,dd_matrix[dd_edges[0], dd_edges[1]])
        x_dis_f2 = torch.relu(x_dis_f2)
        # [88,128]

        X_cir = torch.cat((x_cir_f1, x_cir_f2), 1).t()
        # [256,585]
        X_cir = X_cir.view(1, 2,128, -1)
        #[1, 2, 128, 585]

        X_dis = torch.cat((x_dis_f1, x_dis_f2), 1).t()
        # [256,88]
        X_dis = X_dis.view(1, 2, 128, -1)
        #[1, 2, 128, 88]

        cir_fea = self.cnn_cir(X_cir)
        #[1, 256, 1, 585]
        cir_fea = cir_fea.view(256, 585).t()
        #[585,256]

        dis_fea = self.cnn_dis(X_dis)
        #[1, 256, 1, 88]
        dis_fea = dis_fea.view(256, 88).t()
        # [88,256]

        # [585, 88]
        return cir_fea.mm(dis_fea.t()), cir_fea, dis_fea

class GraphCDANoGAT(nn.Module):
    def __init__(self):
        super(GraphCDANoGAT,self).__init__()
        self.gcn_cir_f1 = GCNConv(128, 128)
        # self.gat_cir_f1 = GATConv(128, 128, 8, concat=False, edge_dim=1)
        self.gcn_cir_f2 = GCNConv(128, 128)

        self.gcn_dis_f1 = GCNConv(128, 128)
        # self.gat_dis_f1 = GATConv(128, 128, 8, concat=False, edge_dim=1)
        self.gcn_dis_f2 = GCNConv(128, 128)

        self.cnn_cir = nn.Conv2d(2, 256, kernel_size=(128, 1), stride=1, bias=True)
        self.cnn_dis = nn.Conv2d(2, 256, kernel_size=(128, 1), stride=1, bias=True)

    def forward(self,x):
        x_cir=torch.randn(585,128).to(device)
        x_dis=torch.randn(88,128).to(device)

        cc_matrix=x['cc_matrix'].to(device)
        cc_edges = x['cc_edges'].to(device)
        dd_matrix = x['dd_matrix'].to(device)
        dd_edges = x['dd_edges'].to(device)

        x_cir_f1=self.gcn_cir_f1(x_cir,cc_edges,cc_matrix[cc_edges[0],cc_edges[1]])
        x_cir_f1=F.relu(x_cir_f1)
        # x_cir_att = self.gat_cir_f1(x_cir_f1, cc_edges,cc_matrix[cc_edges[0], cc_edges[1]])
        # x_cir_att = F.relu(x_cir_att)
        x_cir_f2 = self.gcn_cir_f2(x_cir_f1, cc_edges,cc_matrix[cc_edges[0], cc_edges[1]])
        x_cir_f2 = F.relu(x_cir_f2)
        #[585,128]

        x_dis_f1 = self.gcn_dis_f1(x_dis, dd_edges,dd_matrix[dd_edges[0], dd_edges[1]])
        x_dis_f1 = F.relu(x_dis_f1)
        # x_dis_att = self.gat_dis_f1(x_dis_f1, dd_edges,dd_matrix[dd_edges[0], dd_edges[1]])
        # x_dis_att = F.relu(x_dis_att)
        x_dis_f2 = self.gcn_dis_f2(x_dis_f1, dd_edges,dd_matrix[dd_edges[0], dd_edges[1]])
        x_dis_f2 = F.relu(x_dis_f2)
        # [88,128]

        X_cir = torch.cat((x_cir_f1, x_cir_f2), 1).t()
        # [256,585]
        X_cir = X_cir.view(1, 2,128, -1)
        #[1, 2, 128, 585]

        X_dis = torch.cat((x_dis_f1, x_dis_f2), 1).t()
        # [256,88]
        X_dis = X_dis.view(1, 2, 128, -1)
        #[1, 2, 128, 88]

        cir_fea = self.cnn_cir(X_cir)
        #[1, 256, 1, 585]
        cir_fea = cir_fea.view(256, 585).t()
        #[585,256]

        dis_fea = self.cnn_dis(X_dis)
        #[1, 256, 1, 88]
        dis_fea = dis_fea.view(256, 88).t()
        # [88,256]

        # [585, 88]
        return cir_fea.mm(dis_fea.t()), cir_fea, dis_fea

class GraphCDALast(nn.Module):
    def __init__(self):
        super(GraphCDALast,self).__init__()
        self.gcn_cir_f1 = GCNConv(128, 128)
        # self.gat_cir_f1 = GATConv(128, 128, heads=head_nums, concat=False, edge_dim=1)
        self.gcn_cir_f2 = GCNConv(128, 128)

        self.gcn_dis_f1 = GCNConv(128, 128)
        # self.gat_dis_f1 = GATConv(128, 128, heads=head_nums, concat=False, edge_dim=1)
        self.gcn_dis_f2 = GCNConv(128, 128)

        # self.cnn_cir = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=(128, 1), stride=(1, 1), bias=True)
        # self.cnn_dis = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=(128, 1), stride=(1, 1), bias=True)

    def forward(self,x):
        torch.manual_seed(1)
        x_cir=torch.randn(585,128).to(device)
        x_dis=torch.randn(88,128).to(device)

        cc_matrix=x['cc_matrix'].to(device)
        cc_edges = x['cc_edges'].to(device)
        dd_matrix = x['dd_matrix'].to(device)
        dd_edges = x['dd_edges'].to(device)

        x_cir_f1=self.gcn_cir_f1(x_cir,cc_edges,cc_matrix[cc_edges[0],cc_edges[1]])
        x_cir_f1=torch.relu(x_cir_f1)
        # x_cir_att = self.gat_cir_f1(x_cir_f1, cc_edges,cc_matrix[cc_edges[0], cc_edges[1]])
        # x_cir_att = torch.relu(x_cir_att)
        x_cir_f2 = self.gcn_cir_f2(x_cir_f1, cc_edges,cc_matrix[cc_edges[0], cc_edges[1]])
        x_cir_f2 = torch.relu(x_cir_f2)
        #[585,128]

        x_dis_f1 = self.gcn_dis_f1(x_dis, dd_edges,dd_matrix[dd_edges[0], dd_edges[1]])
        x_dis_f1 = torch.relu(x_dis_f1)
        # x_dis_att = self.gat_dis_f1(x_dis_f1, dd_edges,dd_matrix[dd_edges[0], dd_edges[1]])
        # x_dis_att = torch.relu(x_dis_att)
        x_dis_f2 = self.gcn_dis_f2(x_dis_f1, dd_edges,dd_matrix[dd_edges[0], dd_edges[1]])
        x_dis_f2 = torch.relu(x_dis_f2)
        # [88,128]

        X_cir = torch.cat((x_cir_f1, x_cir_f2), 1).t()
        # [256,585]
        X_cir = X_cir.view(1, 2,128, -1)
        #[1, 2, 128, 585]

        X_dis = torch.cat((x_dis_f1, x_dis_f2), 1).t()
        # [256,88]
        X_dis = X_dis.view(1, 2, 128, -1)
        #[1, 2, 128, 88]

        # cir_fea = self.cnn_cir(X_cir)
        #[1, 256, 1, 585]
        cir_fea = X_cir.view(256, 585).t()
        #[585,256]

        # dis_fea = self.cnn_dis(X_dis)
        #[1, 256, 1, 88]
        dis_fea = X_dis.view(256, 88).t()
        # [88,256]

        # [585, 88]
        return cir_fea.mm(dis_fea.t()), cir_fea, dis_fea



class MyModel1(nn.Module):
    def __init__(self):
        super(MyModel1,self).__init__()

        self.gcn_cir_f1 = GCNConv(128, 128)
        self.gat_cir_f1 = RGCNConv(128, 128,  num_relations=4,num_bases=2)
        # self.gat_cir_f1 = RGATConv(128, 128,heads=4, num_relations=1, num_bases=2)
        self.gcn_cir_f2 = GCNConv(128, 128)
        # self.gcn_cir_f2 = GATConv(128, 128, heads=4, concat=False, edge_dim=1)

        self.gcn_dis_f1 = GCNConv(128, 128)
        self.gat_dis_f1 = RGCNConv(128, 128, num_relations=4, num_bases=2)
        # self.gat_dis_f1 = RGATConv(128, 128, heads=4, num_relations=1,num_bases=2)
        self.gcn_dis_f2 = GCNConv(128, 128)
        # self.gcn_dis_f2 = GATConv(128, 128, heads=4, concat=False, edge_dim=1)

        self.cnn_cir = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=(128, 1), stride=(1, 1), bias=True)
        self.cnn_dis = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=(128, 1), stride=(1, 1), bias=True)

    def forward(self,x):
        torch.manual_seed(1)
        x_cir=torch.randn(585,128).to(device)
        x_dis=torch.randn(88,128).to(device)

        cc_matrix=x['cc_matrix'].to(device)
        cc_edges = x['cc_edges'].to(device)
        dd_matrix = x['dd_matrix'].to(device)
        dd_edges = x['dd_edges'].to(device)

        x_cir_f1=self.gcn_cir_f1(x_cir,cc_edges,cc_matrix[cc_edges[0],cc_edges[1]])
        x_cir_f1=torch.relu(x_cir_f1)
        x_cir_att = self.gat_cir_f1(x_cir_f1, cc_edges,cc_matrix[cc_edges[0], cc_edges[1]])
        x_cir_att = torch.relu(x_cir_att)
        x_cir_f2 = self.gcn_cir_f2(x_cir_att, cc_edges,cc_matrix[cc_edges[0], cc_edges[1]])
        x_cir_f2 = torch.relu(x_cir_f2)
        #[585,128]

        x_dis_f1 = self.gcn_dis_f1(x_dis, dd_edges,dd_matrix[dd_edges[0], dd_edges[1]])
        x_dis_f1 = torch.relu(x_dis_f1)
        x_dis_att = self.gat_dis_f1(x_dis_f1, dd_edges,dd_matrix[dd_edges[0], dd_edges[1]])
        x_dis_att = torch.relu(x_dis_att)
        x_dis_f2 = self.gcn_dis_f2(x_dis_att, dd_edges,dd_matrix[dd_edges[0], dd_edges[1]])
        x_dis_f2 = torch.relu(x_dis_f2)
        # [88,128]

        X_cir = torch.cat((x_cir_f1, x_cir_f2), 1).t()
        # [256,585]
        X_cir = X_cir.view(1, 2,128, -1)
        #[1, 2, 128, 585]

        X_dis = torch.cat((x_dis_f1, x_dis_f2), 1).t()
        # [256,88]
        X_dis = X_dis.view(1, 2, 128, -1)
        #[1, 2, 128, 88]

        cir_fea = self.cnn_cir(X_cir)
        #[1, 256, 1, 585]
        cir_fea = cir_fea.view(256, 585).t()
        #[585,256]

        dis_fea = self.cnn_dis(X_dis)
        #[1, 256, 1, 88]
        dis_fea = dis_fea.view(256, 88).t()
        # [88,256]

        # [585, 88]
        return cir_fea.mm(dis_fea.t()), cir_fea, dis_fea

