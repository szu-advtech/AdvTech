import os

from LSGAN import Generator, Discriminator
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
savepath = "./result/"
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
netG = Generator()
netG = nn.DataParallel(netG).cpu()
netG.load_state_dict(torch.load("./ckpt/lsgan_two_stage/netG_epoch_60.pth",map_location=lambda  storage,loc: storage))
netG.eval()
torch.no_grad()
noise = torch.randn(64, 100, 1, 1, device=device)
fake_images = netG(noise)
vutils.save_image(fake_images.detach()[0:64],
					'%s/fake_samples.png' % (savepath+'lsgan/'), normalize=True)
# for i in range(0,20):
#     noise = torch.randn(64, 100, 1, 1, device=device)
#     fake = netG(noise).detach().cpu()
#     print(fake.shape)
#     rows = vutils.make_grid(fake, padding=2, normalize=True)
#     fig = plt.figure(figsize=(8, 8))
#     plt.imshow(np.transpose(rows, (1, 2, 0)))
#     plt.axis('off')
#     plt.savefig(os.path.join(savepath, "%d.png" % (i)))
#     plt.close(fig)


