import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import *
from models.modules import BasicModel
import numpy as np

class BasicSuper(BasicModel):
    def __init__(self, opt):
        super(BasicSuper, self).__init__(opt)
        if self.mode_supernet in ['all', 'all2', 'feature']:
            self.threshold = self.init_threshold(0)
        if self.mode_supernet in ['all', 'all2', 'embed']:
            self.potential_dim_masks = self.pre_potential_dim_mask()

    def get_random_dim_mask(self, batch_size):
        indexes = torch.randint(low=0, high=self.latent_dim, size=(batch_size, self.field_num)).to(self.device)
        field_masks = F.embedding(indexes, self.potential_dim_masks)
        return field_masks
    
    def calc_sparsity(self):
        base = self.feature_num * self.latent_dim
        params = torch.nonzero(self.calc_feature_mask()).size(0) * self.latent_dim
        percentage = 1 - (params / base)
        return percentage, params

    def calculate_input(self, x, phase):
        xv = F.embedding(x, self.embedding)
        # oov_xv = F.embedding(self.oov_index, self.oov_embedding)
        if self.mode_supernet == 'feature':
            tv = self.get_batch_threshold_value(x)
            if self.mode_oov == 'zero':
                mask_f = self.get_batch_feature_mask(xv, tv)
                xe = torch.mul(mask_f, xv)
            # elif self.mode_oov == 'oov':
            #     mask_f = self.get_batch_feature_mask(xv, tv) # 增1
            #     xe = torch.where(mask_f > 0, xv, oov_xv)
        elif self.mode_supernet == 'embed':
            if phase == 'train':
                mask_e = self.get_random_dim_mask(xv.shape[0])
                xe = torch.mul(mask_e, xv)
            elif phase == 'test':
                xe = xv
        elif self.mode_supernet == 'all':
            tv = self.get_batch_threshold_value(x)
            mask_f = self.get_batch_feature_mask(xv, tv)
            if phase == 'train':
                mask_e = self.get_random_dim_mask(xv.shape[0])
                if self.mode_oov == 'zero':
                    xe = torch.mul(mask_f, torch.mul(mask_e, xv))
                # elif self.mode_oov == 'oov':
                #     xe = torch.where(mask_f > 0, xv, oov_xv)
                #     xe = torch.mul(mask_e, xe)
            elif phase == 'test':
                if self.mode_oov == 'zero':
                    xe = torch.mul(mask_f, xv)
                # elif self.mode_oov == 'oov':
                #     xe = torch.where(mask_f > 0, xv, oov_xv)
        elif self.mode_supernet == 'all2':
            tv = self.get_batch_threshold_value(x)
            if phase == 'train':
                indexes = torch.randint(low=0, high=self.latent_dim, size=(xv.shape[0], self.field_num)).to(self.device)
                mask_e = F.embedding(indexes, self.potential_dim_masks)
                xv = torch.mul(mask_e, xv)
                xv_norm = torch.norm(xv, self.norm, dim=2)
                normalized_xv_norm = torch.div(xv_norm * 64, indexes+1).unsqueeze(2)
                mask_f = self.sigma(normalized_xv_norm - tv)
                if self.mode_oov == 'zero':
                    xe = torch.mul(mask_f, xv)
                # elif self.mode_oov == 'oov':
                #     xe = torch.where(mask_f > 0, xv, oov_xv)
            elif phase == 'test':
                xv_norm = torch.norm(xv, self.norm, dim=2).unsqueeze(2)
                mask_f = self.sigma(xv_norm - tv)
                if self.mode_oov == 'zero':
                    xe = torch.mul(mask_f, xv)
                # elif self.mode_oov == 'oov':
                #     xe = torch.where(mask_f > 0, xv, oov_xv)
        elif self.mode_supernet == 'none':
            #mask_f = torch.ones((xv.shape[0], self.field_num), device=self.device, requires_grad=False).unsqueeze(2)
            #mask_e = torch.ones((xv.shape[0], self.field_num, self.latent_dim), device=self.device, requires_grad=False)
            #xe = torch.mul(mask_f, torch.mul(mask_e, xv))
            xe = xv
        return xe

class FM_super(BasicSuper):
    def __init__(self, opt):
        super(FM_super, self).__init__(opt)
        self.linear = FeaturesLinear(opt['field_dim'])
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x, phase="test"):
        linear_score = self.linear.forward(x)
        xe = self.calculate_input(x, phase)
        fm_score = self.fm.forward(xe)
        score = linear_score + fm_score
        return score.squeeze(1)

class DeepFM_super(FM_super):
    def __init__(self, opt):
        super(DeepFM_super, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout)

    def forward(self, x, phase='test'):
        linear_score = self.linear.forward(x)
        xe = self.calculate_input(x, phase)
        fm_score = self.fm.forward(xe)
        dnn_score = self.mlp.forward(xe.view(-1, self.embed_output_dim))
        score = linear_score + fm_score + dnn_score
        return score.squeeze(1)

class AutoInt_super(DeepFM_super):
    def __init__(self, opt):
        super(AutoInt_super, self).__init__(opt)
        self.has_residual = opt['has_residual']
        self.full_part = opt['full_part']
        self.num_heads = opt['num_heads']
        self.num_layers = opt['num_layers']
        self.atten_embed_dim = opt['atten_embed_dim']
        self.att_dropout = opt['att_dropout']

        self.atten_output_dim = self.field_num * self.atten_embed_dim
        self.dnn_input_dim = self.field_num * self.latent_dim

        self.atten_embedding = torch.nn.Linear(self.latent_dim, self.atten_embed_dim)
        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(self.atten_embed_dim, self.num_heads, dropout=self.att_dropout) for _ in range(self.num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(self.latent_dim, self.atten_embed_dim)

    def forward(self, x, phase='test'):
        xe = self.calculate_input(x, phase)
        score = self.autoint_layer(xe)
        if self.full_part:
            dnn_score = self.mlp.forward(xe.view(-1, self.embed_output_dim))
            score = dnn_score + score

        return score.squeeze(1)

    def autoint_layer(self, xv):
        """Multi-head self-attention layer"""
        atten_x = self.atten_embedding(xv)  # bs, field_num, atten_dim
        cross_term = atten_x.transpose(0, 1)  # field_num, bs, atten_dim
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)  # bs, field_num, atten_dim
        if self.has_residual:
            V_res = self.V_res_embedding(xv)
            cross_term += V_res
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)  # bs, field_num * atten_dim
        output = self.attn_fc(cross_term)
        return output

class FNN_super(BasicSuper):
    def __init__(self, opt):
        super(FNN_super, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout)

    def forward(self, x, phase='test'):
        xv = self.calculate_input(x, phase)
        score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        return score.squeeze(1)

class IPNN_super(BasicSuper):
    def __init__(self, opt):
        super(IPNN_super, self).__init__(opt)      
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

    def forward(self, x, phase='test'):
        xv = self.calculate_input(x, phase)
        product = self.calculate_product(xv)
        xv = xv.view(-1, self.embed_output_dim)
        xe = torch.cat((xv, product), 1)
        score = self.mlp.forward(xe)
        return score.squeeze(1)

class DCN_super(BasicSuper):
    def __init__(self, opt):
        super(DCN_super, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout, output_layer=False)
        self.cross = CrossNetwork(self.embed_output_dim, opt['cross_layer_num'])
        self.combine = torch.nn.Linear(self.mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x, phase='test'):
        xe = self.calculate_input(x, phase)
        dnn_score = self.mlp.forward(xe.view(-1, self.embed_output_dim))
        cross_score = self.cross.forward(xe.view(-1, self.embed_output_dim))
        stacked = torch.cat((dnn_score, cross_score), 1)
        logit = self.combine(stacked)
        return logit.squeeze(1)
