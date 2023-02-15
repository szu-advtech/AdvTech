import torch
from torchviz import make_dot

from models import Model
from models.losses import VGG_Loss, KLD_Loss, GAN_FLoss, GAN_Loss

rank=1
model = Model().to(rank)
vgg_losser = VGG_Loss(rank)
kld_losser = KLD_Loss()
gan_losser = GAN_Loss("hinge")
gan_flosser = GAN_FLoss()

real_img = torch.randn(1,3,256,256).to(rank)
seg = torch.randn(1,2,256,256).to(rank)

mu, log_s = model.encode_img(real_img)
z = model.get_nosie(mu, log_s)
fake_img = model.gen_img(z, seg)
fake_score = model.det_img(fake_img, seg)
real_score = model.det_img(real_img, seg)
kld_loss = kld_losser(mu, log_s)
vgg_loss = vgg_losser(real_img, fake_img)
gan_floss = gan_flosser(real_score, fake_score)
gan_loss = gan_losser(fake_score[-1], True, True)
gen_loss = gan_loss + 0.05*kld_loss + 10*vgg_loss + 10*gan_floss

g = make_dot(gen_loss)
g.render("./graph")