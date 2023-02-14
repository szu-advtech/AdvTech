import math

import torch
import torch.nn as nn
import torch.nn.functional as F
#空间通道特征相关性
def testdis(fm):
    if (fm.size(1) > 64):
        fm = F.avg_pool3d(fm, (4, 1, 1))
    hc = fm.view(fm.size(0), -1, fm.size(2))
    wc = fm.transpose(1,2)
    wc = wc.reshape(wc.size(0), wc.size(1), -1)
    norm_hc = hc / (torch.sqrt(torch.sum(torch.pow(hc, 2), 1)).unsqueeze(1).expand(hc.shape) + 0.0000001)
    norm_wc = wc / (torch.sqrt(torch.sum(torch.pow(wc, 2), 1)).unsqueeze(1).expand(wc.shape) + 0.0000001)
    hchc = norm_hc.bmm(norm_hc.transpose(1, 2))
    wcwc = norm_wc.transpose(1, 2).bmm(norm_wc)
    ans = hchc.unsqueeze(1) + wcwc.unsqueeze(1)
    gamma, P_order = 0.4, 2
    corr_mat = torch.zeros_like(ans)
    for p in range(P_order + 1):
        corr_mat += math.exp(-2 * gamma) * (2 * gamma) ** p / math.factorial(p) * torch.pow(ans, p)
    return corr_mat

    # return ans
    # print("ans",ans.shape)
    # amh = AT(fm)
    # amw = AT(fm.transpose(2,3))
    # amh = amh.view(amh.size(0), -1, 1)
    # amw = amw.view(amw.size(0), -1, 1)
    # am = amh.bmm(amw.transpose(1, 2))
    # print("am",am.shape)
    # return ans.unsqueeze(1)
    # gamma, P_order = 0.4, 2
    # sim_mat = ans.unsqueeze(1) + am.unsqueeze(1)
    # corr_mat = torch.zeros_like(sim_mat)
    # for p in range(P_order + 1):
    #     corr_mat += math.exp(-2 * gamma) * (2 * gamma) ** p / math.factorial(p) * torch.pow(sim_mat, p)
    # return corr_mat
    # gamma, P_order = 0.4, 2

    # am = AT(fm)
    # fm = fm.view(fm.size(0), fm.size(1), -1)
    # norm_ss = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001)
    # am = am.view(am.size(0), am.size(1), -1)
    # am = am.transpose(1, 2).bmm(am)
    # ss = norm_ss.transpose(1, 2).bmm(norm_ss)
    # return ss.unsqueeze(1) + am.unsqueeze(1)

    # sim_mat = ss.unsqueeze(1) + am.unsqueeze(1)
    # corr_mat = torch.zeros_like(sim_mat)
    # for p in range(P_order + 1):
    #     corr_mat += math.exp(-2 * gamma) * (2 * gamma) ** p / math.factorial(p) * torch.pow(sim_mat, p)
    # return corr_mat
#SC
def testdis_old(fm):
    # fm = fm.view(fm.size(0), fm.size(1), -1)
    # norm_ss = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001)
    # norm_cs = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 2)).unsqueeze(2).expand(fm.shape) + 0.0000001)
    # ss = norm_ss.transpose(1, 2).bmm(norm_ss)
    # cs = norm_cs.transpose(1, 2).bmm(norm_cs)
    # s = ss.unsqueeze(1) + cs.unsqueeze(1)
    # return s

    gamma, P_order = 0.4, 2
    # fm = fm.view(fm.size(0), fm.size(1), -1)
    # norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 2)).unsqueeze(2).expand(fm.shape) + 0.0000001)
    # s = norm_fm.transpose(1, 2).bmm(norm_fm)
    # sim_mat = s.unsqueeze(1)
    fm = fm.view(fm.size(0), fm.size(1), -1)
    # print("fm",fm.shape)
    norm_ss = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001)
    # norm_cs = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 2)).unsqueeze(2).expand(fm.shape) + 0.0000001)
    # am = am / (torch.norm(torch.sum(torch.pow(torch.abs(fm), 2), dim=1, keepdim=True), dim=(2, 3), keepdim=True) + 1e-6)
    ss = norm_ss.transpose(1, 2).bmm(norm_ss)
    # cs = norm_cs.transpose(1, 2).bmm(norm_cs)
    # print("ss", ss.shape, "cs", cs.shape)
    sim_mat = ss.unsqueeze(1)
    # print("simmat", sim_mat.shape)
    corr_mat = torch.zeros_like(sim_mat)
    for p in range(P_order + 1):
        corr_mat += math.exp(-2 * gamma) * (2 * gamma) ** p / math.factorial(p) * torch.pow(sim_mat, p)
        # print("corrmat",corr_mat.shape)
    return corr_mat

# 通道相关性
def channel_similarity(fm):
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 2)).unsqueeze(2).expand(fm.shape) + 0.0000001)
    if (norm_fm.size(1) < 256):
        norm_fm = norm_fm.view(1, norm_fm.size(0) * norm_fm.size(1), norm_fm.size(2))
    s = norm_fm.bmm(norm_fm.transpose(1, 2))
    # s = norm_fm.transpose(1, 2).bmm(norm_fm)
    s = s.unsqueeze(1)
    return s

def spatial_similarity(fm):
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001)
    s = norm_fm.transpose(1, 2).bmm(norm_fm)
    s = s.unsqueeze(1)
    return s

#事例相关性
def batch_similarity(fm):
    fm = fm.view(fm.size(0), -1)
    Q = torch.mm(fm, fm.transpose(0, 1))
    normalized_Q = Q / torch.norm(Q, 2, dim=1).unsqueeze(1).expand(Q.shape)
    return normalized_Q

def FSP(fm1, fm2):
    if fm1.size(2) > fm2.size(2):
        fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

    fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
    fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)

    # fsp = torch.bmm(fm1, fm2) / fm1.size(2)
    fsp = fm2.bmm(fm1) / fm1.size(2)
    return fsp
#AT算法
def AT(fm):
    eps=1e-6
    am = torch.pow(torch.abs(fm), 2)
    # am = torch.sum(am, dim=1, keepdim=True)
    # norm = torch.norm(am, dim=(2,3), keepdim=True)
    am = torch.sum(am, dim=3, keepdim=True)
    norm = torch.norm(am, dim=(1, 2), keepdim=True)
    am = torch.div(am, norm+eps)
    return am
#SP算法
def SP(fm):
    fm = fm.view(fm.size(0), -1)
    G = torch.mm(fm, fm.t())
    norm_G = F.normalize(G, p=2, dim=1)
    return norm_G
#CC算法
def CC(fm1, fm2):
    gamma, P_order = 0.4, 2
    fm1 = fm1.view(fm1.size(0), -1)
    fm2 = fm2.view(fm2.size(0), -1)
    fm1 = F.normalize(fm1, p=2, dim=-1)
    fm2 = F.normalize(fm2, p=2, dim=-1)
    sim_mat = torch.matmul(fm1, fm2.t())
    corr_mat = torch.zeros_like(sim_mat)
    for p in range(P_order+1):
        corr_mat += math.exp(-2*gamma) * (2*gamma)**p / math.factorial(p) * torch.pow(sim_mat, p)
    return corr_mat

def FT(fm):
    norm_factor = F.normalize(fm.view(fm.size(0), -1))
    return norm_factor
