import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import *
from models.modules import BasicModel
import numpy as np

class BasicEvo(BasicModel):
    def __init__(self, opt):
        super(BasicEvo, self).__init__(opt)
        if self.mode_supernet == 'all':
            self.threshold = self.init_threshold()
        self.potential_dim_masks = self.pre_potential_dim_mask()

    def prepare_sparse_feature(self):
        if self.mode_oov == 'zero':
            self.sparse_embedding = torch.mul(self.embedding, self.calc_feature_mask().unsqueeze(1))
        self.feature_mask = self.calc_feature_mask()

    def calc_sparsity(self, cand=None):
        feature_mask = self.feature_mask.cpu().detach().numpy()
        base = self.feature_num * self.latent_dim
        if cand is None:
            params = np.sum(feature_mask) * self.latent_dim
        else:
            offset = 0
            params = 0
            for i, num_i in enumerate(self.field_dim):
                f_i = np.sum(feature_mask[offset:offset+num_i])
                params += f_i * cand[i]
                offset += num_i
        percentage = 1 - (params / base)
        return percentage, int(params)

    def calculate_input(self, x, cand):
        if self.mode_oov == 'zero':
            xe = F.embedding(x, self.sparse_embedding)
        elif self.mode_oov == 'oov':
            xv = F.embedding(x, self.embedding)
            oov_xv = F.embedding(self.oov_index, self.oov_embedding)
            mask_f = F.embedding(x, self.feature_mask).unsqueeze(2)
            xe = torch.where(mask_f > 0, xv, oov_xv)
        mask_e = F.embedding(cand, self.potential_dim_masks)
        xe = torch.mul(mask_e, xe)
        return xe

class FM_evo(BasicEvo):
    def __init__(self, opt):
        super(FM_evo, self).__init__(opt)
        self.linear = FeaturesLinear(opt['field_dim'])  # linear part
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x, cand):
        linear_score = self.linear.forward(x)
        xv = self.calculate_input(x, cand)
        fm_score = self.fm.forward(xv)
        score = linear_score + fm_score
        return score.squeeze(1)

class DeepFM_evo(FM_evo):
    def __init__(self, opt):
        super(DeepFM_evo, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout)

    def forward(self, x, cand):
        linear_score = self.linear.forward(x)
        xv = self.calculate_input(x, cand)
        fm_score = self.fm.forward(xv)
        dnn_score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        score = linear_score + fm_score + dnn_score
        return score.squeeze(1)

class FNN_evo(BasicEvo):
    def __init__(self, opt):
        super(FNN_evo, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout)

    def forward(self, x, cand):
        xv = self.calculate_input(x, cand)
        score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        return score.squeeze(1)

class IPNN_evo(BasicEvo):
    def __init__(self, opt):
        super(IPNN_evo, self).__init__(opt)      
        self.embed_output_dim = self.field_num * self.latent_dim
        self.product_output_dim = int(self.field_num * (self.field_num - 1) / 2)
        self.dnn_input_dim = self.embed_output_dim + self.product_output_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.dnn_input_dim, self.mlp_dims, dropout=self.dropout)

        # Create indexes
        rows = []
        cols = []
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                rows.append(i)
                cols.append(j)
        self.rows = torch.tensor(rows, device=self.device)
        self.cols = torch.tensor(cols, device=self.device)

    def calculate_product(self, xe):
        batch_size = xe.shape[0]
        trans = torch.transpose(xe, 1, 2)
        gather_rows = torch.gather(trans, 2, self.rows.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        gather_cols = torch.gather(trans, 2, self.cols.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        p = torch.transpose(gather_rows, 1, 2)
        q = torch.transpose(gather_cols, 1, 2)
        product_embedding = torch.mul(p, q)
        product_embedding = torch.sum(product_embedding, 2)
        return product_embedding

    def forward(self, x, cand):
        xv = self.calculate_input(x, cand)
        product = self.calculate_product(xv)
        xv = xv.view(-1, self.embed_output_dim)
        xe = torch.cat((xv, product), 1)
        score = self.mlp.forward(xe)
        return score.squeeze(1)

class DCN_evo(BasicEvo):
    def __init__(self, opt):
        super(DCN_evo, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout, output_layer=False)
        self.cross = CrossNetwork(self.embed_output_dim, opt['cross_layer_num'])
        self.combine = torch.nn.Linear(self.mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x, cand):
        xv = self.calculate_input(x, cand)
        dnn_score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        cross_score = self.cross.forward(xv.view(-1, self.embed_output_dim))
        stacked = torch.cat((dnn_score, cross_score), 1)
        logit = self.combine(stacked)
        return logit.squeeze(1)
