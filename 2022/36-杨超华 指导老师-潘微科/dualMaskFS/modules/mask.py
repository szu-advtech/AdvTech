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
        self.mask_weight = nn.Parameter(torch.Tensor(self.feature_num, 1))
        nn.init.constant_(self.mask_weight, self.mask_initial_value)
    
    def compute_mask(self, x, temp, ticket):
        scaling = 1./ sigmoid(self.mask_initial_value)
        mask_weight = F.embedding(x, self.mask_weight)
        if ticket:
            mask = (mask_weight > 0).float()
        else:
            mask = torch.sigmoid(temp * mask_weight)
        return scaling * mask
    
    def prune(self, temp):
        self.mask_weight.data = torch.clamp(temp * self.mask_weight.data, max=self.mask_initial_value)


    def forward(self, x, temp=1, ticket=False):
        embed = F.embedding(x, self.embedding)
        mask = self.compute_mask(x, temp, ticket)
        return embed * mask
    
    def compute_remaining_weights(self, temp, ticket=False):
        if ticket:
            return float((self.mask_weight > 0.).sum()) / self.mask_weight.numel()
        else:
            m = torch.sigmoid(temp * self.mask_weight)
            print("max mask weight: {wa:6f}, min mask weight: {wi:6f}".format(wa=torch.max(self.mask_weight),wi=torch.min(self.mask_weight)))
            print("max mask: {ma:8f}, min mask: {mi:8f}".format(ma=torch.max(m), mi=torch.min(m)))
            print("mask number: {mn:6f}".format(mn=float((m==0.).sum())))
            return 1 - float((m == 0.).sum()) / m.numel()

    def checkpoint(self):
        self.init_weight.data = self.embedding.clone()
    
    def rewind_weights(self):
        self.embedding.data = self.init_weight.clone()

    def reg(self, temp):
        return torch.sum(torch.sigmoid(temp * self.mask_weight))


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

    def reg(self):
        reg_loss = 0.
        for m in self.mask_modules:
            reg_loss += m.reg(self.temp)
        return reg_loss


class MaskDNN(MaskedNet):
    def __init__(self, opt):
        super(MaskDNN, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding = self.mask_embedding(x, self.temp, self.ticket)
        # output_linear = self.linear(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_dnn = self.dnn(x_dnn)
        logit = output_dnn
        return logit

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
    weight_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask_weight' not in p[0], network.named_parameters()))
    mask_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask_weight' in p[0], network.named_parameters()))
    optim = optim.lower()
    if optim == "sgd":
        return [torch.optim.SGD(weight_params, lr=lr, weight_decay=l2), torch.optim.SGD(mask_params, lr=lr)]
    elif optim == "adam":
        return [torch.optim.Adam(weight_params, lr=lr, weight_decay=l2), torch.optim.Adam(mask_params, lr=lr)]
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
