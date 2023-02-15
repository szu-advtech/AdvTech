import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import *
from models.modules import BasicModel
import numpy as np
import time

class BasicRetrain(BasicModel):
    def __init__(self, opt):
        super(BasicRetrain, self).__init__(opt)
        self.mode_retrain = opt['mode_retrain']
        if self.mode_retrain == 'feature':
            self.threshold = self.init_threshold(0)

    def update_mask(self, embed_mask=None, feature_mask=None):
        # Embed mask
        if embed_mask is None:
            embed_mask = torch.ones((self.field_num,), dtype=torch.int64, device=self.device)
            embed_mask = torch.mul(embed_mask, self.latent_dim) - 1
        total_masks = self.pre_potential_dim_mask()
        self.embed_mask = F.embedding(embed_mask, total_masks)
        self.embed_mask.requires_grad_(False)

        # Feature mask
        if feature_mask is None:
            feature_mask = torch.ones((self.feature_num,), device=self.device)
        self.feature_mask = feature_mask
        self.feature_mask.requires_grad_(False)

    def calc_sparsity(self):
        if self.mode_retrain == 'feature':
            feature_mask = self.calc_feature_mask().cpu().numpy()
        elif self.mode_retrain == 'weight':
            feature_mask = self.feature_mask.cpu().numpy()
        embed_mask = self.embed_mask.cpu().numpy().sum(axis=1)
        base = self.feature_num * self.latent_dim
        offset = 0
        params = 0
        for i, num_i in enumerate(self.field_dim):
            f_i = np.sum(feature_mask[offset:offset+num_i])
            params += f_i * embed_mask[i]
            offset += num_i
        percentage = 1 - (params / base)
        return percentage, int(params)

    def calculate_input(self, x):
        xv = F.embedding(x, self.embedding)
        if self.mode_retrain == 'feature':
            tv = self.get_batch_threshold_value(x)
            mask_f = self.get_batch_feature_mask(xv, tv)
        elif self.mode_retrain == 'weight':
            mask_f = F.embedding(x, self.feature_mask).unsqueeze(2)
        if self.mode_oov == 'zero':
            xe = torch.mul(mask_f, xv)
        elif self.mode_oov == 'oov':
            oov_xv = F.embedding(self.oov_index, self.oov_embedding)
            xe = torch.where(mask_f > 0, xv, oov_xv)
        mask_e = self.embed_mask
        xe = torch.mul(mask_e, xe)
        return xe

class FM_retrain(BasicRetrain):
    def __init__(self, opt):
        super(FM_retrain, self).__init__(opt)
        self.linear = FeaturesLinear(opt['field_dim'])
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        linear_score = self.linear.forward(x)
        xv = self.calculate_input(x)
        fm_score = self.fm.forward(xv)
        score = linear_score + fm_score
        return score.squeeze(1)

class DeepFM_retrain(FM_retrain):
    def __init__(self, opt):
        super(DeepFM_retrain, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout)

    def forward(self, x):
        linear_score = self.linear.forward(x)
        xv = self.calculate_input(x)
        fm_score = self.fm.forward(xv)
        dnn_score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        score = linear_score + fm_score + dnn_score
        return score.squeeze(1)

    def calc_forward_time(self, x):
        mask_f = F.embedding(x, self.feature_mask).unsqueeze(2)
        mask_e = self.embed_mask
        t1 = time.time()
        linear_score = self.linear.forward(x)
        xv = F.embedding(x, self.embedding)
        xe = torch.mul(mask_f, xv)
        xe = torch.mul(mask_e, xe)
        fm_score = self.fm.forward(xe)
        dnn_score = self.mlp.forward(xe.view(-1, self.embed_output_dim))
        score = linear_score + fm_score + dnn_score
        t2 = time.time()
        return t2-t1

class FNN_retrain(BasicRetrain):
    def __init__(self, opt):
        super(FNN_retrain, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout)

    def forward(self, x):
        xv = self.calculate_input(x)
        score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        return score.squeeze(1)

class IPNN_retrain(BasicRetrain):
    def __init__(self, opt):
        super(IPNN_retrain, self).__init__(opt)      
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

    def forward(self, x):
        xv = self.calculate_input(x)
        product = self.calculate_product(xv)
        xv = xv.view(-1, self.embed_output_dim)
        xe = torch.cat((xv, product), 1)
        score = self.mlp.forward(xe)
        return score.squeeze(1)

class DCN_retrain(BasicRetrain):
    def __init__(self, opt):
        super(DCN_retrain, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout, output_layer=False)
        self.cross = CrossNetwork(self.embed_output_dim, opt['cross_layer_num'])
        self.combine = torch.nn.Linear(self.mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        xe = self.calculate_input(x)
        dnn_score = self.mlp.forward(xe.view(-1, self.embed_output_dim))
        cross_score = self.cross.forward(xe.view(-1, self.embed_output_dim))
        stacked = torch.cat((dnn_score, cross_score), 1)
        logit = self.combine(stacked)
        return logit.squeeze(1)
