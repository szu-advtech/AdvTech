import torch
import torch.nn as nn


class My_loss(nn.Module):
    def __init__(self):
        super(My_loss,self).__init__()  # 没有需要保存的参数和状态信息

    def forward(self, S_o,S_gt,S_u,mask,X_o,X_gt,X_final,X_u):  # 定义前向的函数运算即可

        L1 = torch.norm(S_o - S_gt,p=1).cuda() + torch.norm(S_u - S_gt,p=1).cuda()

        L2 = torch.norm((1-mask) * (X_o - X_gt),p=1).cuda()


        L3 = torch.norm((1 - mask) * (X_final - X_gt), p=1).cuda() + torch.norm(
            (1 - mask) * (X_u - X_gt), p=1).cuda()

        L_tot = (5 * L1 + L2 + L3).cuda()
        return L_tot