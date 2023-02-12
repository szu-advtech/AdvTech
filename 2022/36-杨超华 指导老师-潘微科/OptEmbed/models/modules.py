import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import *

class BasicModel(torch.nn.Module):
    def __init__(self, opt):
        super(BasicModel, self).__init__()
        self.device = torch.device("cuda:0" if opt.get('use_cuda') else "cpu")
        self.latent_dim = opt['latent_dim']
        self.field_num = len(opt['field_dim'])
        self.feature_num = sum(opt['field_dim'])
        self.field_dim = opt['field_dim']
        self.mode_supernet = opt["mode_supernet"]
        self.mode_threshold = opt['mode_threshold']
        self.mode_oov = opt['mode_oov']
        assert self.mode_supernet in ['feature', 'embed', 'none', 'all', 'all2']
        assert self.mode_threshold in ['feature', 'field']
        assert self.mode_oov in ['oov', 'zero']
        self.sigma = BinaryStep.apply
        self.norm = opt['norm']
        assert self.norm in [1, 2]

        self.embedding = self.init_embedding()
        print("BackBone Embedding Parameters: ", self.feature_num * self.latent_dim)

        # Add OOV Embedding Table
        self.oov_embedding = self.init_oov_embedding()
        self.oov_index = torch.IntTensor([i for i in range(self.field_num)]).to(self.device)

    def init_embedding(self):
        e = nn.Parameter(torch.rand([self.feature_num, self.latent_dim]))
        torch.nn.init.xavier_uniform_(e)
        return e

    def init_oov_embedding(self):
        oov_e = nn.Parameter(torch.rand([self.field_num, self.latent_dim]))
        torch.nn.init.xavier_uniform_(oov_e)
        return oov_e

    def init_threshold(self, t_init=0):
        if self.mode_threshold == 'feature':
            t = nn.Parameter(torch.empty(self.feature_num, 1))
        elif self.mode_threshold == 'field':
            t = nn.Parameter(torch.empty(self.field_num, 1))
        nn.init.constant_(t, -t_init)
        return t

    def pre_potential_dim_mask(self):
        masks = []
        for i in range(self.latent_dim):
            zeros = np.zeros(self.latent_dim - i - 1)
            ones = np.ones(i + 1)
            mask = torch.from_numpy(np.concatenate((ones, zeros), axis=0)).unsqueeze(0)
            masks.append(mask)
        total_masks = torch.cat(masks, dim=0).float().to(self.device)
        return total_masks

    def calc_feature_mask(self):
        if self.mode_supernet in ['all', 'feature']:
            embedding_norm = torch.norm(self.embedding, self.norm, dim=1, keepdim=True)
            if self.mode_threshold in ['feature']:
                final_threshold = self.threshold
            elif self.mode_threshold in ['field']:
                k = []
                for i, dim in enumerate(self.field_dim):
                    ti = self.threshold[i]
                    ki = ti.expand(dim)
                    k.append(ki)
                final_threshold = torch.cat(k, dim=0).unsqueeze(1)
            # feature_mask = torch.where(self.sigma(embedding_norm - final_threshold) > 0, 1, 0)
            feature_mask = self.sigma(embedding_norm - final_threshold)
            return feature_mask.squeeze()
        elif self.mode_supernet in ['embed', 'none']:
            return torch.ones(self.feature_num, device=self.device, requires_grad=False)

    def get_batch_feature_mask(self, xv, tv):
        xv_norm = torch.norm(xv, self.norm, dim=2).unsqueeze(2)
        mask_f = self.sigma(xv_norm - tv)
        return mask_f        

    def get_batch_threshold_value(self, x):
        if self.mode_threshold == 'feature':
            tv = F.embedding(x, self.threshold)
        elif self.mode_threshold == 'field':

            tv = self.threshold
        return tv
