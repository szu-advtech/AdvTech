import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from dataset import testDeepLesionImageDataset
from model import DuDoUFNet
import cv2
import numpy as np
from loss import My_loss
from FBP import voxel_backprojection,siddon_ray_projection



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
def Normalize(data):
    mins, maxs = torch.min(data), torch.max(data)
    img_nor = (data - mins) / (maxs  - mins )
    return img_nor

type = 'dose2'
h5_dir = r'/home/wsy/matlabProject/DuDoNet/database_MAR/images_test/%s/'%(type)

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
test_data = testDeepLesionImageDataset('h5_%s_test_list.txt'%(type), h5_dir, )
train_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
model = DuDoUFNet(2,3,16)
optimizer = optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=1e-4)
total_loss = My_loss()
total_data = len(test_data)
print('测试样本数量：%d'%(total_data))
model.load_state_dict(torch.load("/home/wsy/pythonProject/UFNet_dose2/%s/saveModel/Best/bestmodel_%s.pth"%(type,type)))
model.cuda()
model.eval()

ssim_sum,ssim_sum2, psnr_sum,psnr_sum2, mse_sum, mse_sum2 = 0,0,0,0,0,0
sum = 0
with torch.no_grad():
    for ii, data in enumerate(train_dataloader):
        sum += 1
        print(sum)

        X_ldma, X_gt, M, idx  = [x.cuda() for x in data]
        S_ldma = fan_fp(X_ldma)
        M_proj = fan_fp(M)
        fbpX_ldma = fan_bp(S_ldma)

        S_o, S_u, X_final, X_u, X_o = model(S_ldma, M_proj, X_ldma, M)
        # X_o = (1-M)*X_o
        metal_X = ((1 - M) * X_gt + M).cuda()
        metal_X = Normalize(metal_X)
        X_final = Normalize(X_final)
        fbpX_ldma = Normalize(fbpX_ldma)

        mse = (torch.sum((metal_X * 255 - X_final * 255) ** 2) / X_final.numel()).cuda()
        mse2 = (torch.sum((metal_X * 255 - fbpX_ldma * 255) ** 2) / X_final.numel()).cuda()
        psnr = (10 * torch.log10(255 ** 2 / mse)).cuda()
        psnr2 = (10 * torch.log10(255 ** 2 / mse2)).cuda()
        ssim_rgb = ssim(metal_X.cpu().numpy()[0, 0, :, :] * 255, X_final.detach().cpu().numpy()[0, 0, :, :] * 255)
        ssim_rgb2 = ssim(metal_X.cpu().numpy()[0, 0, :, :] * 255, fbpX_ldma.cpu().numpy()[0, 0, :, :] * 255)
        ssim_sum += ssim_rgb
        psnr_sum += psnr.data.item()
        mse_sum += mse.data.item()

        ssim_sum2 += ssim_rgb2
        psnr_sum2 += psnr2.data.item()
        mse_sum2 += mse2.data.item()


total_data = sum
print('FBP: PSNR:{:.3f}\t SSIM:{:.3f}\t RMSE:{:.3f}'.format(psnr_sum2 / total_data, ssim_sum2/total_data, np.sqrt(mse_sum2/total_data) ))
print('DuDoUFNet: PSNR:{:.3f}\t SSIM:{:.3f}\t RMSE:{:.3f}'.format(psnr_sum / total_data, ssim_sum/total_data, np.sqrt(mse_sum/total_data) ))


with open("/home/wsy/pythonProject/UFNet/test_result/%s/average_test_score.txt"%(type),'w') as f:
    f.write('FBP: PSNR:{:.3f}\t SSIM:{:.3f}\t RMSE:{:.3f}'.format(psnr_sum2 / total_data, ssim_sum2/total_data, np.sqrt(mse_sum2/total_data) ) + '\n')
    f.write('DuDoUFNet: PSNR:{:.3f}\t SSIM:{:.3f}\t RMSE:{:.3f}'.format(psnr_sum / total_data, ssim_sum/total_data, np.sqrt(mse_sum/total_data) ) + '\n')