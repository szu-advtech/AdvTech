import torch


class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')
    # logits是网络预测的HatMap
    # targets是GT的HatMap
    def __call__(self, logits, targets):
        # [bs,c,H,w]
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]

        weight = torch.tensor((0.02,0.98)).to(device)

        # [B, num_kps, H, W] -> [B, num_kps] 在H和W两个维度上取均值
        loss = self.criterion(logits, targets).mean(dim=[2, 3])
        loss = torch.sum(loss * weight) / bs
        return loss
