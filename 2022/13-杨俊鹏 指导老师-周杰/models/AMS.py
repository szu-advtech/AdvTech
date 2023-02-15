# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, margin=0.35, scale=30, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.margin = margin
        self.scale = scale

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def unitize_weight(self):
        norm_w = torch.norm(self.weight, 2, 1, keepdim=True)
        unit_w = self.weight/(norm_w.clamp(min=1e-4))
        self.weight.data = unit_w.data

    def forward(self, input, target=None):
        norm_w = torch.norm(self.weight, p=2, dim=1, keepdim=True)
        unit_w = self.weight/(norm_w.clamp(min=1e-4))
        norm_x = torch.norm(input, p=2, dim=1, keepdim=True)
        unit_x = input/(norm_x.clamp(min=1e-4))
        wx = unit_x.matmul(unit_w.t())
        wx = wx.clamp(-1, 1) # for numerical stability
        if target is not None:
            #norm_x = torch.norm(input, 2, 1).view(-1, 1)
            cos_t = wx
            cos_mt = cos_t - self.margin
            cos_t_mt = cos_t

            # mask = cos_t.data.gt(2.0) # cos\theta can not be greater than 2.0
            # a = torch.ByteTensor(1)
            # a[0] = 1
            # for i in range(target.size(0)):
            #     index = target[i].data
            #     mask[i][index] = a.cuda()
            with torch.no_grad():
                mask = cos_t.new_zeros(cos_t.shape, dtype=torch.uint8).cuda()
                mask.scatter_(1, target.view(-1, 1), 1)
            # mask = Variable(mask)
            cos_t_mt.masked_scatter_(mask, torch.masked_select(cos_mt, mask))
            wx = cos_t_mt
        return wx * self.scale

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features)  \
            + ', margin=' + str(self.margin) \
            + ', scale=' + str(self.scale) + ')'


class CosFaceWithAdaptiveMargin(nn.Module):
    def __init__(self, in_features, out_features, max_margin=0.4, scale=16, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.max_margin = max_margin
        self.scale = scale

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def unitize_weight(self):
        norm_w = torch.norm(self.weight, 2, 1, keepdim=True)
        unit_w = self.weight/norm_w
        self.weight.data = unit_w.data

    def forward(self, input, acc=0.0, target=None):
        norm_w = torch.norm(self.weight, p=2, dim=1, keepdim=True)
        unit_w = self.weight/norm_w
        norm_x = torch.norm(input, p=2, dim=1, keepdim=True)
        unit_x = input/norm_x
        wx = unit_x.matmul(unit_w.t())
        wx = wx.clamp(-1, 1) # for numerical stability
        if target is not None:
            # use Sigmoid Function to adjust margin
            # margin = self.max_margin * (1.0 / 1 + math.exp(-6*acc+4))
            cos_t = wx
            cos_mt = cos_t - margin
            cos_t_mt = cos_t

            with torch.no_grad():
                mask = cos_t.new_zeros(cos_t.shape, dtype=torch.uint8).cuda() # cos\theta can not be greater than 2.0
                # a = torch.ByteTensor(1)
                # a[0] = 1
                # print(mask.shape)
                # for i in range(target.size(0)):
                #     index = target[i].data
                #     mask[i][index] = a.cuda()
                mask.scatter_(1, target.view(-1, 1), 1)

            #mask.requires_grad = True
            cos_t_mt.masked_scatter_(mask, torch.masked_select(cos_mt, mask))
            wx = cos_t_mt
        return wx * self.scale

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features)  \
            + ', margin=' + str(self.margin) \
            + ', scale=' + str(self.scale) + ')'

class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, in_features, out_features, s = 64.0, m = 0.50, easy_margin = False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        #nn.init.xavier_uniform_(self.kernel)
        nn.init.normal_(self.kernel, std=0.01)
        self.register_parameter('bias', None)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embbedings, label):
        #embbedings = l2_norm(embbedings, axis = 1)
        #kernel_norm = l2_norm(self.kernel, axis = 0)
        embbedings = F.normalize(embbedings, p=2, dim=1)
        kernel_norm = F.normalize(self.kernel, p=2, dim=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)

        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output , origin_cos * self.s


    def l2_norm(input, axis = 1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)

        return output

class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        #embbedings = l2_norm(embbedings, axis = 1)
        #kernel_norm = l2_norm(self.kernel, axis = 0)
        embbedings = F.normalize(embbedings, p=2, dim=1)
        kernel_norm = F.normalize(self.kernel, p=2, dim=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s


if __name__ == '__main__':
    x = torch.randn(8, 256, requires_grad=True)
    label = torch.tensor([0,1,2,3,4,5,6,7],dtype=torch.long)
    arcface = ArcFace(256, 8)
    y = arcface(x, label)
    crit = nn.CrossEntropyLoss()
    loss = crit(y, label)
    loss.backward()