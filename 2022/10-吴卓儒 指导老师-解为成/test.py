import argparse
import time
import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
# 加载gc模块
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from data_loader import load_real_data,generate_real_data,resize,generate_fake_data_coarse,generate_fake_data_fine
from model_pytorch import Novel_Residual_Block,Coarse_Generator,Fine_Generator,Discriminator,Fundus2Angio,Fundus2Angio_Gan,freeze,unfreeze

def data_normalize_trans_tensor(input):
    #cuda = True if torch.cuda.is_available() else False
    #T = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    T = torch.FloatTensor
    # normalize from [0,255] to [-1,1]
    input = (input - 127.5) / 127.5
    input_trans = np.transpose(input,(0,3,1,2))
    input_tensor = torch.from_numpy(input_trans).type(T)
    return input_tensor

test_dataset = load_real_data('fun2angio.npz') #args.npz_file+'.npz'
print('Loaded', test_dataset[0].shape, test_dataset[1].shape)

print(len(test_dataset[0]))

# define generator models
coarse_generator = Coarse_Generator(input_nc=3)

fine_generator = Fine_Generator(input_nc=3)

cg_state_path = "weights_plots/coarse_generator_model_021200.pth"
fg_state_path = "weights_plots/fine_generator_model_021200.pth"

coarse_generator.load_state_dict(torch.load(cg_state_path))
fine_generator.load_state_dict(torch.load(fg_state_path))

for batch in tqdm.tqdm(range(len(test_dataset[0]))):
    src, tar =test_dataset[0][batch], test_dataset[1][batch]
    # print(src)
    # print(src.shape)
    # print(tar)
    # print(tar.shape)

    src = np.expand_dims(src, axis=0)
    tar = np.expand_dims(tar, axis=0)
    # Resize to half
    out_shape = (int(src.shape[1] / 2), int(src.shape[2] / 2))
    # print(out_shape) (256,256)
    [src_half, tar_half] = resize(src, tar, out_shape)
    src = data_normalize_trans_tensor(src)
    tar = data_normalize_trans_tensor(tar)
    src_half = data_normalize_trans_tensor(src_half)
    tar_half = data_normalize_trans_tensor(tar_half)
    # print(src)
    # print(src.shape)
    # print(tar)
    # print(tar.shape)

    fake_half, feature = coarse_generator(src_half)

    fake = fine_generator((src, feature))

    # print(fake)
    # print(fake.shape)

    fake_tensor = fake.clone()
    fake_numpy = fake_tensor.detach().numpy()

    # recover normalize from [-1,1] to [0,255]
    fake_numpy = fake_numpy * 127.5 + 127.5

    # print(fake_numpy)
    # print(fake_numpy.shape)

    fake_numpy = fake_numpy[0,0,:,:]

    im = Image.fromarray(fake_numpy.astype(np.uint8))
    if not os.path.exists("test_results"):
        os.makedirs("test_results_epo100")
    filename = "test_results_epo100/"+str(batch+1)+"_synthesized_image.png"
    im.save(filename)
    print('>Saved: %s' % filename)


