import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from numpy.linalg import norm
import sys,os
import torch.nn.functional as F
#from Common.Const import GPU
from torch.autograd import Variable, grad
from torch.autograd import Variable
from torch.distributions import Beta
from numpy import ones,zeros
import functools

def dist_o2l(p1, p2):
    # distance from origin to the line defined by (p1, p2)
    p12 = p2 - p1
    u12 = p12 / np.linalg.norm(p12)
    l_pp = np.dot(-p1, u12)
    pp = l_pp*u12 + p1
    return np.linalg.norm(pp)

def para_count(models):
    count = 0
    for model in models:
        count +=  sum(param.numel() for param in model.parameters())
    return count

class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=True):
        super(CrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, preds, gts):
        gts = gts.contiguous().view(-1)

        if self.smoothing:
            eps = 0.2
            n_class = preds.size(1)

            one_hot = torch.zeros_like(preds).scatter(1, gts.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(preds, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(preds, gts, reduction='mean')

        return loss


def smooth_labels(B,ran=[0.9,1.0]):
    #return y - 0.3 + (np.random.random(y.shape) * 0.5)
    return (ran[1]-ran[0])*np.random.random(B) + ran[0]

def noisy_labels(y, p_flip=0.05):
    # determine the number of labels to flip
    n_select = int(p_flip * y.shape[0])
    # choose labels to flip
    flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select)
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    return y

def BCEloss(D_fake, D_real, d_real_target, d_fake_target):
    real =  F.binary_cross_entropy_with_logits(D_real,d_real_target.expand_as(D_real))
    fake =  F.binary_cross_entropy_with_logits(D_fake,d_fake_target.expand_as(D_fake))
    return real, fake

def dis_loss(d_real, d_fake, gan="wgan", weight=1.,d_real_p=None, d_fake_p=None, noise_label=False):
    # B = d_fake.size(0)
    # a = 1.0
    # b = 0.9

    if gan.lower() == "wgan":
        loss_fake = d_fake.mean()
        loss_real = d_real.mean()
        wg_loss_orig = loss_fake - loss_real
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_dis_loss": wg_loss.clone().detach().item(),
            "wgan_dis_loss_orig": wg_loss_orig.clone().detach().item(),
            "wgan_dis_loss_real": loss_real.clone().detach().item(),
            "wgan_dis_loss_fake": loss_fake.clone().detach().item()
        }
    elif gan.lower() == "hinge":
        d_loss_real = torch.nn.ReLU()(1.0 - d_real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + d_fake).mean()

        # d_loss_real = -torch.min(d_real - 1, d_real * 0).mean()
        # d_loss_fake = -torch.min(-d_fake - 1, d_fake * 0).mean()
        real_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        real_acc = real_correct / float(d_real.size(0) + d_fake.size(0))

        d_loss = d_loss_real + d_loss_fake
        loss = d_loss * weight
        return loss, {
            "loss": loss.clone().detach(),
            "d_loss": d_loss.clone().detach(),
            "dis_acc": real_acc.clone().detach(),
            "dis_correct": real_correct.clone().detach(),
            "loss_real": d_loss_real.clone().detach(),
            "loss_fake": d_loss_fake.clone().detach(),
        }
    elif gan.lower() == "ls":
        mse = nn.MSELoss()
        B = d_fake.size(0)

        real_label_np = np.ones((B,))
        fake_label_np = np.zeros((B,))

        if noise_label:
            real_label_np = smooth_labels(B,ran=[0.9,1.0])
            #fake_label_np = smooth_labels(B,ran=[0.0,0.1])
            # occasionally flip the labels when training the D to
            # prevent D from becoming too strong
            real_label_np = noisy_labels(real_label_np, 0.05)
            #fake_label_np = noisy_labels(fake_label_np, 0.05)


        real_label = torch.from_numpy(real_label_np.astype(np.float32)).cuda()
        fake_label = torch.from_numpy(fake_label_np.astype(np.float32)).cuda()


        # real_label = Variable((1.0 - 0.9) * torch.rand(d_fake.size(0)) + 0.9).cuda()
        # fake_label = Variable((0.1 - 0.0) * torch.rand(d_fake.size(0)) + 0.0).cuda()

        t = 0.5
        real_correct = (d_real >= t).float().sum()
        real_acc = real_correct / float(d_real.size(0))

        fake_correct  = (d_fake < t).float().sum()
        fake_acc = fake_correct / float(d_fake.size(0))
        # + d_fake.size(0))

        # real_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(1).cuda())
        # fake_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(0).cuda())

        g_loss = F.mse_loss(d_fake, fake_label)
        d_loss = F.mse_loss(d_real, real_label)

        if d_real_p is not None and d_fake_p is not None:

            real_label_p = Variable((1.0 - 0.9) * torch.rand(d_fake_p.size(0), d_fake_p.size(1)) + 0.9).cuda()
            fake_label_p = Variable((0.1 - 0.0) * torch.rand(d_fake_p.size(0), d_fake_p.size(1)) + 0.0).cuda()

            # real_label_p = Variable(torch.FloatTensor(d_real_p.size(0), d_real_p.size(1)).fill_(1).cuda())
            # fake_label_p = Variable(torch.FloatTensor(d_real_p.size(0), d_real_p.size(1)).fill_(0).cuda())
            g_loss_p = F.mse_loss(d_fake_p, fake_label_p)
            d_loss_p = F.mse_loss(d_real_p, real_label_p)

            g_loss = (g_loss + 0.1*g_loss_p)
            d_loss = (d_loss + 0.1*d_loss_p)

        loss =  (g_loss+d_loss)/2.0
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach(),
            'd_loss': g_loss.clone().detach(),
            "fake_acc": fake_acc.clone().detach(),
            "real_acc": real_acc.clone().detach()
        }
    elif gan.lower() =="gan":
        d_real_target = torch.tensor([1.0]).cuda()
        d_fake_target = torch.tensor([0.0]).cuda()
        discriminator_loss = functools.partial(BCEloss, d_real_target=d_real_target, d_fake_target=d_fake_target)

        g_loss, d_loss = discriminator_loss(d_fake, d_real)

        if d_real_p is not None and d_fake_p is not None:
            g_loss_p,d_loss_p = discriminator_loss(d_fake_p.view(-1),d_real_p.view(-1))
            g_loss = (g_loss + g_loss_p)/2.0
            d_loss = (d_loss + d_loss_p)/2.0

        loss =  (g_loss+d_loss)/2.0
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach(),
            'd_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "real":
        y = Variable(torch.Tensor(d_real.size(0)).fill_(1.0), requires_grad=False)
        d_loss = torch.mean((d_real - torch.mean(d_fake) - y) ** 2)
        g_loss = torch.mean((d_fake - torch.mean(d_real) + y) ** 2)
        loss =  (g_loss+d_loss)/2.0

    else:
        raise NotImplementedError("Not implement: %s" % gan)

def BCEfakeloss(D_fake,target):
    return F.binary_cross_entropy_with_logits(D_fake, target.expand_as(D_fake))


def gen_loss(d_real, d_fake, gan="wgan", weight=1., d_real_p=None, d_fake_p=None,noise_label=False):
    if gan.lower() == "wgan":
        wg_loss_orig = - d_fake.mean()
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_gen_loss": wg_loss.clone().detach().item(),
            "wgan_gen_loss_orig": wg_loss_orig.clone().detach().item(),
        }
    elif gan.lower() == "hinge":
        g_loss = -d_fake.mean()
        d_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        d_acc = d_correct / float(d_real.size(0) + d_fake.size(0))

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            "dis_acc": d_acc.clone().detach(),
            "dis_correct": d_correct.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "ls":
        #mse = nn.MSELoss()
        B = d_fake.size(0)
        #real_label_np = np.ones((B,))
        fake_label_np = np.ones((B,))

        if noise_label:
            # occasionally flip the labels when training the generator to fool the D
            fake_label_np = noisy_labels(fake_label_np, 0.05)

        #real_label = torch.from_numpy(real_label_np.astype(np.float32)).cuda()

        fake_label = torch.from_numpy(fake_label_np.astype(np.float32)).cuda()

        # real_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(1).cuda())
        # fake_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(0).cuda())
        g_loss = F.mse_loss(d_fake, fake_label)



        if d_fake_p is not None:
            fake_label_p = Variable(torch.FloatTensor(d_fake_p.size(0), d_fake_p.size(1)).fill_(1).cuda())
            g_loss_p = F.mse_loss(d_fake_p,fake_label_p)
            g_loss = g_loss + 0.2*g_loss_p

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "gan":
        fake_target = torch.tensor([1.0]).cuda()
        fake_loss = functools.partial(BCEfakeloss, target=fake_target)
        g_loss = fake_loss(d_fake)

        if d_fake_p is not None:
            g_loss_p = fake_loss(d_fake_p.view(-1))
            g_loss = g_loss + g_loss_p

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "real":
        # https://github.com/weishenho/SAGAN-with-relativistic/blob/master/main.py
        y = Variable(torch.Tensor(d_real.size(0)).fill_(1.0), requires_grad=False)
        d_loss =  torch.mean((d_real - torch.mean(d_fake) + y) ** 2)
        g_loss = torch.mean((d_fake - torch.mean(d_real) - y) ** 2)

        # d_loss = torch.mean((d_real - torch.mean(d_fake) - y) ** 2)
        # g_loss = torch.mean((d_fake - torch.mean(d_real) + y) ** 2)
        loss = (g_loss + d_loss) / 2.0

    else:
        raise NotImplementedError("Not implement: %s" % gan)