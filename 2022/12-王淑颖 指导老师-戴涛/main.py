import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from dataset import DeepLesionImageDataset
from model import DuDoUFNet
import numpy as np
import logging
import time
from loss import My_loss

import cv2
from FBP import voxel_backprojection,siddon_ray_projection
import matplotlib.pyplot as plt
print(torch.cuda.current_device())

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = 'train_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)

    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log
def Normalize(data):
    mins, maxs = torch.min(data), torch.max(data)
    img_nor = (data - mins) / (maxs  - mins )
    return img_nor
logger = log_creater('./dose4/Logs/')

h5_dir = r'/home/wsy/matlabProject/DuDoNet/database_MAR/images_train/dose4/'

geo_mode = 'fanflat'  # or 'parallel'
angle_range = {'fanflat': 2 * np.pi, 'parallel': np.pi}
geo_full = {'nVoxelX': 416, 'sVoxelX': 340.0192, 'dVoxelX': 0.6641,
                    'nVoxelY': 416, 'sVoxelY': 340.0192, 'dVoxelY': 0.6641,
                    'nDetecU': 640, 'sDetecU': 504.0128, 'dDetecU': 0.6848,
                    'views': 640, 'slices': 1,
                    'DSD': 600.0, 'DSO': 550.0, 'DOD': 50.0,
                    'start_angle': 0.0, 'end_angle': angle_range[geo_mode],
                    'mode': geo_mode
                    }
fan_bp = voxel_backprojection(geo_full)
fan_fp = siddon_ray_projection(geo_full)
training_data = DeepLesionImageDataset('h5_dose4_train_list.txt', h5_dir, )
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
logger.info('dose4训练样本数量：%d'%len(training_data))
logger.info('DuDoUFNet(2,3,16)')

model = DuDoUFNet(2,3,16).cuda()
optimizer = optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=1e-4)
total_loss = My_loss()
epochs = 500
flag = 0
best_model,best_PSNR,best_SSIM = 0,0,0
save_gapNum = 0

for epoch in range(epochs):
    loss_per_epoch = 0
    psnr_per_epoch = 0
    ssim_per_epoch = 0
    mse_per_epoch = 0
    for ii, data in enumerate(train_dataloader):
        # S_ldma, X_o,X_ldma, S_gt, X_gt, M, M_proj = [x.cuda() for x in data]
        X_ldma, X_gt, M = [x.cuda() for x in data]
        # fp
        S_ldma = fan_fp(X_ldma)
        # S_ldma = Normalize(S_ldma)
        S_gt = S_ldma
        M_proj = fan_fp(M)
        # M_proj = Normalize(M_proj)
        # X_o = (1-M) * X_o
        model.train()
        optimizer.zero_grad()
        # stage1_inputs = torch.cat((S_ldma, M_proj), 1).cuda()
        S_o, S_u, X_final, X_u, X_o = model(S_ldma,M_proj,X_ldma,M)

        metal_X = ((1-M)*X_gt + M).cuda()
        # FBP
        # reco_img = fan_bp(S_ldma)


        l = total_loss( S_o,M_proj,S_u,M,X_o,X_gt,X_final,X_u)
        l.backward()
        optimizer.step()

        loss_per_epoch += l.data.item()

        metal_X = Normalize(metal_X)
        X_final = Normalize(X_final)

        mse = (torch.sum((metal_X*255 - X_final*255)**2) / X_final.numel()).cuda()
        psnr = (10 * torch.log10(255**2 / mse )).cuda()

        ssim_rgb = ssim(metal_X.cpu().numpy()[0,0,:,:]*255,X_final.detach().cpu().numpy()[0,0,:,:]*255)
        psnr_per_epoch += psnr.data.item()
        ssim_per_epoch += ssim_rgb
        mse_per_epoch += mse.data.item()
        if ii % 200 == 0:
            logger.info("epoch: {} ,batch id : {} PSNR:{:.8f}, SSIM:{:.8f}, MSE:{:.8f}".format(epoch + 1, ii + 1,psnr.data.item(),ssim_rgb,mse.data.item()))
        # Linit = torch.norm((1-mask) * ())
    average_psnr = psnr_per_epoch / (ii + 1)
    average_ssim = ssim_per_epoch / (ii + 1)

    logger.info('{}, {},  {}/{}, total_loss: {:.8f},PSNR:{:.8f}, SSIM:{:.8f}, MSE:{:.8f}'.format('DuDoUFNet', 'train', epoch,
                                                        epochs, loss_per_epoch / (ii + 1), average_psnr,
                                                        average_ssim,mse_per_epoch/(ii + 1)))

    save_gapNum += 1
    if save_gapNum >= 20:# 结束训练
        break

    if best_model < ((average_ssim*100 + average_psnr) / 2.0):
        best_model = (average_ssim*100 + average_psnr) / 2.0
        #保存模型
        torch.save(model.state_dict(),'.//dose4/saveModel/Best/bestmodel_dose4.pth')
        save_gapNum = 0
        logger.info('====================保存Best模型=======================')

