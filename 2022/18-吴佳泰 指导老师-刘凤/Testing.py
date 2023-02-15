import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from lib.PraNet_Res2Net import PraNet
from dataloader import test_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-19.pth')


data_path = './data/Test_Data/'
save_path = './results/result/'
opt = parser.parse_args()
model = PraNet()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

os.makedirs(save_path, exist_ok=True)
image_root = '{}/Testing_data/'
gt_root = '{}/GroundTruth/'
test_loader = test_dataset(image_root, gt_root, opt.testsize)

for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()

    res5, res4, res3, res2 = model(image)
    res = res2
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res=255*res
    res=res.astype(np.uint8)
    cv2.imwrite(save_path+name, res)