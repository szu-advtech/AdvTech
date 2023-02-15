import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.layers import FactorizationMachine, MultiLayerPerceptron
import copy
import modules.layers as layer


class MaskEmbedding(nn.Module):
    def __init__(self, feature_num, latent_dim, mask_initial_value=0.):
        super().__init__()
        self.feature_num = feature_num
        self.latent_dim = latent_dim
        self.mask_initial_value = mask_initial_value
        self.embedding = nn.Parameter(torch.zeros(feature_num, latent_dim))
        nn.init.xavier_uniform_(self.embedding)
        self.init_weight = nn.Parameter(torch.zeros_like(self.embedding), requires_grad=False)
        self.init_mask()
    
    def init_mask(self):
        self.mask_weight_i = nn.Parameter(torch.Tensor(self.feature_num, 1))
        self.mask_weight_s = nn.Parameter(torch.Tensor(self.feature_num, 1))
        self.mask_weight_j = nn.Parameter(torch.Tensor(self.feature_num, 1))
        nn.init.constant_(self.mask_weight_i, self.mask_initial_value)
        nn.init.constant_(self.mask_weight_s, self.mask_initial_value)
        nn.init.constant_(self.mask_weight_j, self.mask_initial_value)
    
    def compute_mask(self, x, temp, ticket):
        scaling = 1./ sigmoid(self.mask_initial_value)
        mask_weight_i = F.embedding(x, self.mask_weight_i)
        mask_weight_s = F.embedding(x, self.mask_weight_s)
        mask_weight_j = F.embedding(x, self.mask_weight_j)
        if ticket:
            mask_i = (mask_weight_i > 0).float()
            mask_s = (mask_weight_s > 0).float()
            mask_j = (mask_weight_j > 0).float()
        else:
            mask_i = torch.sigmoid(temp * mask_weight_i)
            mask_s = torch.sigmoid(temp * mask_weight_s)
            mask_j = torch.sigmoid(temp * mask_weight_j)
        # 1
        #return scaling * mask_i, scaling * mask_s, scaling * mask_j
        # 2
        return 0.5 * scaling * mask_i, 0.5 * scaling * mask_s, 0.5 * scaling * mask_j
    
    def prune(self, temp):
        self.mask_weight_i.data = torch.clamp(temp * self.mask_weight_i.data, max=self.mask_initial_value)
        self.mask_weight_s.data = torch.clamp(temp * self.mask_weight_s.data, max=self.mask_initial_value)
        self.mask_weight_j.data = torch.clamp(temp * self.mask_weight_j.data, max=self.mask_initial_value)


    def forward(self, x, temp=1, ticket=False):
        embed = F.embedding(x, self.embedding)
        mask_i, mask_s, mask_j = self.compute_mask(x, temp, ticket)
        # 1
        # mask1 = mask_i + mask_s
        # mask2 = mask_j + mask_s
        # 2
        # mask1 = torch.clamp(mask_i + mask_s, max=1.0)
        # mask2 = torch.clamp(mask_j + mask_s, max=1.0)
        # 3
        g_s = torch.where(mask_s < 0.5,  torch.Tensor([1]).to(torch.device('cpu')), torch.Tensor([0]).to(torch.device('cpu')))
        mask1 = mask_i * g_s + mask_s * (1 - g_s)
        mask2 = mask_j * g_s + mask_s * (1 - g_s)
        return embed * mask1, embed * mask2
    
    def compute_remaining_weights(self, temp, ticket=False):
        if ticket:
            return float((self.mask_weight_s > 0.).sum()) / self.mask_weight_s.numel(), float((self.mask_weight_i > 0.).sum()) / self.mask_weight_i.numel(), float((self.mask_weight_j > 0.).sum()) / self.mask_weight_j.numel()
        else:
            m_s = torch.sigmoid(temp * self.mask_weight_s)
            m_i = torch.sigmoid(temp * self.mask_weight_i)
            m_j = torch.sigmoid(temp * self.mask_weight_j)
            print("max mask weight s: {wa:6f}, min mask weight s: {wi:6f}".format(wa=torch.max(self.mask_weight_s),wi=torch.min(self.mask_weight_s)))
            print("max mask s: {ma:8f}, min mask s: {mi:8f}".format(ma=torch.max(m_s), mi=torch.min(m_s)))
            print("mask s number: {mn:6f}".format(mn=float((m_s==0.).sum())))
            print("max mask weight i: {wa:6f}, min mask weight i: {wi:6f}".format(wa=torch.max(self.mask_weight_i),wi=torch.min(self.mask_weight_i)))
            print("max mask i: {ma:8f}, min mask i: {mi:8f}".format(ma=torch.max(m_i), mi=torch.min(m_i)))
            print("mask i number: {mn:6f}".format(mn=float((m_i == 0.).sum())))
            print("max mask weight j: {wa:6f}, min mask weight j: {wi:6f}".format(wa=torch.max(self.mask_weight_j),wi=torch.min(self.mask_weight_j)))
            print("max mask j: {ma:8f}, min mask j: {mi:8f}".format(ma=torch.max(m_j), mi=torch.min(m_j)))
            print("mask j number: {mn:6f}".format(mn=float((m_j == 0.).sum())))
            return 1 - float((m_s == 0.).sum()) / m_s.numel(), 1 - float((m_i == 0.).sum()) / m_i.numel(), 1 - float((m_j == 0.).sum()) / m_j.numel()

    def checkpoint(self):
        self.init_weight.data = self.embedding.clone()
    
    def rewind_weights(self):
        self.embedding.data = self.init_weight.clone()

    def reg1_s(self, temp):
        return torch.sum(torch.sigmoid(temp * self.mask_weight_s))

    def reg1_i(self, temp):
        return torch.sum(torch.sigmoid(temp * self.mask_weight_i))

    def reg1_j(self, temp):
        return torch.sum(torch.sigmoid(temp * self.mask_weight_j))

    def reg2(self, temp):
        # 方式1
        return torch.sum(torch.sigmoid(temp * self.mask_weight_i) * torch.sigmoid(temp * self.mask_weight_j))



class MaskedNet(nn.Module):
    def __init__(self, opt):
        super(MaskedNet, self).__init__()
        self.ticket = False
        self.latent_dim = opt["latent_dim"]
        self.feature_num = opt["feat_num"]
        self.field_num = opt["field_num"]
        self.mask_embedding = MaskEmbedding(self.feature_num, self.latent_dim, mask_initial_value=opt["mask_initial"])
        self.mask_modules = [m for m in self.modules() if type(m) == MaskEmbedding]
        self.temp = 1

    def checkpoint(self):
        for m in self.mask_modules: m.checkpoint()
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules: m.rewind_weights()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
                m.load_state_dict(m.checkpoint)
                
    def prune(self):
        for m in self.mask_modules: m.prune(self.temp)

    def reg1_s(self):
        reg_loss = 0.
        for m in self.mask_modules:
            reg_loss += m.reg1_s(self.temp)
        return reg_loss

    def reg1_i(self):
        reg_loss = 0.
        for m in self.mask_modules:
            reg_loss += m.reg1_i(self.temp)
        return reg_loss

    def reg1_j(self):
        reg_loss = 0.
        for m in self.mask_modules:
            reg_loss += m.reg1_j(self.temp)
        return reg_loss

    def reg2(self):
        reg_loss = 0.
        for m in self.mask_modules:
            reg_loss += m.reg2(self.temp)
        return reg_loss



class MaskDNN(MaskedNet):
    def __init__(self, opt):
        super(MaskDNN, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn1 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.dnn2 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding1, x_embedding2 = self.mask_embedding(x, self.temp, self.ticket)
        # output_linear = self.linear(x)
        x_dnn1 = x_embedding1.view(-1, self.dnn_dim)
        x_dnn2 = x_embedding2.view(-1, self.dnn_dim)
        output_dnn1 = self.dnn1(x_dnn1)
        output_dnn2 = self.dnn2(x_dnn2)
        # logit = output_dnn
        return output_dnn1, output_dnn2

    def compute_remaining_weights(self):
        return self.mask_embedding.compute_remaining_weights(self.temp, self.ticket)

class MaskDeepFM(MaskedNet):
    def __init__(self, opt):
        super(MaskDeepFM, self).__init__(opt)
        self.fm = FactorizationMachine(reduce_sum=True)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num*self.latent_dim
        self.dnn = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding = self.mask_embedding(x, self.temp, self.ticket)
        #output_linear = self.linear(x)
        output_fm = self.fm(x_embedding)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_dnn = self.dnn(x_dnn)
        logit = output_dnn + output_fm
        return logit
    
    def compute_remaining_weights(self):
        return self.mask_embedding.compute_remaining_weights(self.temp, self.ticket)


class MaskDeepCross(MaskedNet):
    def __init__(self, opt):
        super(MaskDeepCross, self).__init__(opt)
        self.dnn_dim = self.field_num * self.latent_dim
        cross_num = opt["cross"]
        mlp_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.cross = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination = nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)

    def forward(self, x):
        x_embedding = self.mask_embedding(x, self.temp, self.ticket)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_cross = self.cross(x_dnn)
        output_dnn = self.dnn(x_dnn)
        comb_tensor = torch.cat((output_cross, output_dnn), dim=1)
        logit = self.combination(comb_tensor)
        return logit

    def compute_remaining_weights(self):
        return self.mask_embedding.compute_remaining_weights(self.temp, self.ticket)


class MaskedFM(MaskedNet):
    def __init__(self, opt):
        super(MaskedFM, self).__init__(opt)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_embedding = self.mask_embedding(x, self.temp, self.ticket)
        output_fm = self.fm(x_embedding)
        logits = output_fm
        return logits

    def compute_remaining_weights(self):
        return self.mask_embedding.compute_remaining_weights(self.temp, self.ticket)


class MaskedIPNN(MaskedNet):
    def __init__(self, opt):
        super(MaskedIPNN, self).__init__(opt)
        mlp_dims = opt["mlp_dims"]
        use_bn = opt["use_bn"]
        dropout = opt["mlp_dropout"]
        self.dnn_dim = self.field_num * self.latent_dim + int(self.field_num * (self.field_num - 1) / 2)
        self.inner = layer.InnerProduct(self.field_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=True, dropout=dropout, use_bn=use_bn)
        
    def forward(self, x):
        x_embedding = self.mask_embedding(x)
        x_dnn = x_embedding.view(-1, self.field_num*self.latent_dim)
        x_innerproduct = self.inner(x_embedding)
        x_dnn = torch.cat((x_dnn, x_innerproduct), 1)
        logit = self.dnn(x_dnn)
        return logit

    def compute_remaining_weights(self):
        return self.mask_embedding.compute_remaining_weights(self.temp, self.ticket)


def getOptim(network, optim, lr, l2):
    #w = list(filter(lambda p: p[1].requires_grad and 'mask_weight_s' in p[0], network.named_parameters()))
    weight_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask_weight' not in p[0], network.named_parameters()))
    mask_i_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask_weight_i' in p[0], network.named_parameters()))
    mask_j_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask_weight_j' in p[0], network.named_parameters()))
    mask_s_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask_weight_s' in p[0], network.named_parameters()))
    optim = optim.lower()
    if optim == "sgd":
        return [torch.optim.SGD(weight_params, lr=lr, weight_decay=l2), torch.optim.SGD(mask_i_params, lr=lr), torch.optim.SGD(mask_j_params, lr=lr), torch.optim.SGD(mask_s_params, lr=lr)]
    elif optim == "adam":
        return [torch.optim.Adam(weight_params, lr=lr, weight_decay=l2), torch.optim.Adam(mask_i_params, lr=lr), torch.optim.Adam(mask_j_params, lr=lr), torch.optim.Adam(mask_s_params, lr=lr)]
    else:
        raise ValueError("Invalid optimizer type: {}".format(optim))


def getModel(model:str, opt):
    model = model.lower()
    if model == "deepfm":
        return MaskDeepFM(opt)
    elif model == "dcn":
        return MaskDeepCross(opt)
    elif model == "dnn":
        return MaskDNN(opt)
    elif model == "fm":
        return MaskedFM(opt)
    elif model == "ipnn":
        return MaskedIPNN(opt)
    else:
        raise ValueError("Invalid model type: {}".format(model))


def sigmoid(x):
    return float(1./(1.+np.exp(-x)))
