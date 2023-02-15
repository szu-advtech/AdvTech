import torch
# from models.vgg import VGG19
from torchvision.models import vgg19

class KLD_Loss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, mu, log_s):
        return -0.5 * torch.sum(1+log_s-mu.pow(2)-log_s.exp())

class GAN_Loss(torch.nn.Module):
    def __init__(self, gan_mode) -> None:
        super().__init__()
        self.gan_mode = gan_mode
        self.real_label = None
        self.fake_label = None

    def get_target_label(self, score, flag):
        if flag:
            if self.real_label is None or self.real_label.shape != score.shape:
                self.real_label = torch.ones_like(score)
                self.real_label.requires_grad_(False)
            return self.real_label
        else:
            if self.fake_label is None or self.fake_label.shape != score.shape:
                self.fake_label = torch.zeros_like(score)
                self.fake_label.requires_grad_(False)
            return self.fake_label
        
    def loss(self, score, flag, for_gan=False):
        cur_target = self.get_target_label(score, flag)
        loss = None
        if self.gan_mode == "bce":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(score, cur_target)
        elif self.gan_mode == "mse":
            loss = torch.nn.functional.mse_loss(score, cur_target)
        elif self.gan_mode == "hinge":
            if for_gan:
                loss = -torch.mean(score)
            else:
                if flag:
                    minval = torch.min(score - 1, torch.zeros_like(score))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-score - 1, torch.zeros_like(score))
                    loss = -torch.mean(minval)
        else:
            print("Unknowen gan mode!")
        return loss

    def forward(self, input, flag, for_gan=False):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, flag, for_gan)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, flag, for_gan)

class VGG_Loss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg_features = vgg19(weights='DEFAULT').features.eval()
        self.blocks = torch.nn.ModuleList()
        self.blocks.append(vgg_features[:2])
        self.blocks.append(vgg_features[2:7])
        self.blocks.append(vgg_features[7:12])
        self.blocks.append(vgg_features[12:21])
        self.blocks.append(vgg_features[21:30])
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False

        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, real, fake):
        x = fake
        y = real
        loss = 0.0
        cur = 0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += self.weights[cur]*torch.nn.functional.l1_loss(x, y.detach())
            cur += 1
        return loss

class GAN_FLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.losser = torch.nn.L1Loss()
    def forward(self, real_score, fake_score):
        gan_floss = 0
        n = len(fake_score)-1
        for i in range(n):
            gan_floss += self.losser(fake_score[i], real_score[i].detach())/n
        return gan_floss
