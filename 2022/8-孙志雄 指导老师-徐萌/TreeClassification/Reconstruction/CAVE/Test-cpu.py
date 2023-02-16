"""
@Name: Test-cpu.py
@Auth: SniperIN_IKBear
@Date: 2022/11/22-11:04
@Desc: 
@Ver : 0.0.0
"""
import hdf5storage
import torch
import torch.utils.data as tud
import argparse

from CAVE.Model import HSI_Fusion
from Utils import *
from Dataset_test import cave_dataset
import warnings
import numpy as np
import collections


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='3'

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
parser.add_argument('--data_path', default='/home2/szx/Dataset/TreeDetection/DHIF/Test3/', type=str,
                    help='path of the testing data')
parser.add_argument("--sizeIx", default=2560, type=int, help='the size of trainset')
parser.add_argument("--sizeIy", default=2560, type=int, help='the size of trainset')
parser.add_argument("--testset_num", default=1, type=int, help='total number of testset')
parser.add_argument("--batch_size", default=1, type=int, help='Batch size')
parser.add_argument("--sf", default=8, type=int, help='Scaling factor')
parser.add_argument("--seed", default=1, type=int, help='Random seed')
parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
opt = parser.parse_args()
print(opt)

key = 'Test.txt'
file_path = opt.data_path + key
file_list = loadpath(file_path, shuffle=False)
HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 1)
# 添加維度 RuntimeError: The size of tensor a (9776) must match the size of tensor b (9773) at non-singleton dimension 3

# HR_HSI = HR_HSI[2:, :, :, :]
# temp_zero = np.zeros((1808,3,112,1))
# HR_HSI = np.concatenate([HR_HSI, temp_zero],axis=1)
#
# HR_MSI = HR_MSI[2:, :, :, :]
# temp_zero = np.zeros((1808,3,3,1))
# HR_MSI = np.concatenate([HR_MSI, temp_zero],axis=1)

print(HR_HSI.shape)
print(HR_MSI.shape)

# HR_HSI = data = hdf5.loadmat(Hsi)
# #归一化
#
# HR_HSI = torch.from_numpy(HR_HSI)
# HR_HSI_max = torch.max(HR_HSI)
# HR_HSI_min = torch.min(HR_HSI)
# HR_HSI = (HR_HSI - HR_HSI_min) / (HR_HSI_max - HR_HSI_min)
#
# HR_MSI = torch.from_numpy(HR_MSI)
# HR_MSI_max = torch.max(HR_MSI)
# HR_MSI_min = torch.min(HR_MSI)
# HR_MSI = (HR_MSI - HR_MSI_min) / (HR_MSI_max - HR_MSI_min)
#
# HR_HSI = HR_HSI.numpy()
# HR_MSI = HR_MSI.numpy()

dataset = cave_dataset(opt, HR_HSI, HR_MSI, istrain=False)
loader_train = tud.DataLoader(dataset, batch_size=opt.batch_size)

model = HSI_Fusion(Ch=53, stages=4, sf=opt.sf)
model.load_state_dict(torch.load("./Checkpoint/f8/Model/model_100.pth",map_location=lambda storage, loc: storage).module.state_dict())

# # model = torch.load("./Checkpoint/f8/Model/model_010.pth")
# model = torch.load("./Checkpoint/f8/Model/model_011.pth")
# model.load_state_dict(model)  #加载模型参数，pre_model为直接没有设置
# torch.save(model.state_dict(), './Checkpoint/f8/Model/model_011_cpu.pth',_use_new_zipfile_serialization=False)
#
# model = torch.load("./Checkpoint/f8/Model/model_011_cpu.pth",map_location='cpu')
# model.load_state_dict(model['model'])


# device = torch.device('cpu')
# model = model.to(device)
# set the number of parallel GPUs
print("===> Setting CPU")
# # ------------CPU-----------------
# state_dict_new = collections.OrderedDict()
# for k, v in model.items():
#     name = k[7:]                        # 去掉 "module."
#     state_dict_new[name] = v
# model = model.load_state_dict(state_dict_new)   # 加载模型
# -------------END----------------
model = model.eval()



# model = dataparallel(model, 1)


# model = model.cuda()

psnr_total = 0
k = 0
for j, (LR, RGB, HR) in enumerate(loader_train):
    with torch.no_grad():
        out = model(RGB.cpu(), LR.cpu())
        result = out
        result = result.clamp(min=0., max=1.)
    psnr = compare_psnr(result.cpu().detach().numpy(), HR.numpy(), data_range=1.0)
    psnr_total = psnr_total + psnr
    k = k + 1
    print(psnr)
    #
    res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
    save_path = './Result/ssr/' + str(j + 2) + '.mat'
    print("開始寫入數據")
    hdf5storage.savemat(save_path, {'res': res}, do_compression=True,format='7.3')
    # sio.savemat(save_path, {'res':res})

print(k)
print("Avg PSNR = %.4f" % (psnr_total / k))
