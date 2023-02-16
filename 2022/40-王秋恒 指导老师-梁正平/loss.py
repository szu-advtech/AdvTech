import torch


class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss()
    # logits是网络预测的HatMap
    # targets是GT的HatMap
    def __call__(self, logits, targets):
        # [bs,c,H,w]
        # [B, num_kps, H, W] -> [B, num_kps] 在H和W两个维度上取均值
        loss = self.criterion(logits, targets)
        return loss
