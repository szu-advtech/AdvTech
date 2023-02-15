# In[]
import numpy as np
from modules import Render
import utils
import matplotlib.pyplot as plt
import cv2
import os
from dataset import BaseDataSet
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

DATA_DIR = './results/test'
SAVE_DIR = DATA_DIR+'/rerender'
utils.set_folder(SAVE_DIR)
DATA_DIR = 'D:/data/Data_Deschaintre18/testBlended'
LESS_CHANNEL = False

transform = transforms.Compose([
    transforms.ToTensor()])
test_dataset = BaseDataSet(
    root_dir=DATA_DIR, transform=transform, less_channel=LESS_CHANNEL, split=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1, num_workers=0, shuffle=False)

renderer = Render(less_channel=LESS_CHANNEL)
diffuse_count = 1
specular_count = 3


for j, input in enumerate(test_loader):
    input = torch.cat(input[1:], 1).cuda()
    scenes = utils.generate_diffuse_scenes(
        diffuse_count) + utils.generate_specular_scenes(specular_count, size=input.shape[-2:])
    for i in range(len(scenes)):
        result = renderer(input, **scenes[i])[0]
        tmp = np.asarray(torch.clip(
            utils.tensor_show(result[0], show=False), 0, 1)*255)
        cv2.imwrite(
            os.path.join(SAVE_DIR, 'img_%03d_rerender_%03d.png' % (j, i)),
            cv2.cvtColor(tmp.astype('uint8'), cv2.COLOR_RGB2BGR))
