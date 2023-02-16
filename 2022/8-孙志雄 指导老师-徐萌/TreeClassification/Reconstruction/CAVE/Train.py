import random

import torch

from Model import HSI_Fusion
from CAVE_Dataset import cave_dataset
import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
from torch.autograd import Variable
from Utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    ## Model Config
    parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
    parser.add_argument('--data_path', default='/home2/szx/Dataset/TreeDetection/DHIF/Train/', type=str,
                        help='Path of the training data')
    parser.add_argument("--sizeI", default=96, type=int, help='The image size of the training patches')
    parser.add_argument("--batch_size", default=16, type=int, help='Batch size')
    parser.add_argument("--trainset_num", default=20000, type=int, help='The number of training samples of each epoch')
    parser.add_argument("--sf", default=8, type=int, help='Scaling factor')
    parser.add_argument("--seed", default=1, type=int, help='Random seed')
    parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
    parser.add_argument("--testset_num", default=1, type=int, help='total number of testset')
    opt = parser.parse_args()

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print(opt)

    ## New model
    print("===> New Model")
    model = HSI_Fusion(Ch=53, stages=4, sf=opt.sf)

    ## set the number of parallel GPUs
    print("===> Setting GPU")
    model = dataparallel(model, 3)

    ## Initialize weight
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)

    ## Load training data
    key = 'Train.txt'
    file_path = opt.data_path + key
    file_list = loadpath(file_path)
    HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 1)  ######
    # 归一化

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

    ## Load trained model
    initial_epoch = findLastCheckpoint(save_dir="./Checkpoint/f8/Model")
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join("./Checkpoint/f8/Model", 'model_%03d.pth' % initial_epoch))

    ## Loss function
    criterion = nn.L1Loss()

    ## optimizer and scheduler
    # optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    # scheduler = MultiStepLR(optimizer, milestones=[], gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=list(range(1, 150, 5)), gamma=0.95)

    ## pipline of training
    for epoch in range(initial_epoch, 100):
        model.train()

        dataset = cave_dataset(opt, HR_HSI, HR_MSI)
        loader_train = tud.DataLoader(dataset, num_workers=1, batch_size=opt.batch_size, shuffle=True)

        scheduler.step(epoch)
        epoch_loss = 0

        start_time = time.time()
        for i, (LR, RGB, HR) in enumerate(loader_train):
            LR, RGB, HR = Variable(LR), Variable(RGB), Variable(HR)
            out = model(RGB.cuda(), LR.cuda())

            loss = criterion(out, HR.cuda())
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % int(len(loader_train) / 3) == 0:
                print('%4d %4d / %4d loss = %.10f time = %s' % (
                    epoch + 1, i, len(dataset) // opt.batch_size, epoch_loss / ((i + 1) * opt.batch_size),
                    datetime.datetime.now()))
            ######################################
            # if i > 1:
            #     break
            ######################################
        elapsed_time = time.time() - start_time
        print('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss / len(dataset), elapsed_time))
        torch.save(model, os.path.join("./Checkpoint/f8/Model", 'model_%03d.pth' % (epoch + 1)))  # save model

        ##################Valing###########################




        def val_prepare_data():
            ########################这里需要改变数据####################################
            HR_HSI = np.zeros((((512, 512, 53, 1))))  #######
            HR_MSI = np.zeros((((512, 512, 3, 1))))  #######
            file_id = random.randint(1,49)
            val_path_hsi = '/home2/szx/Dataset/TreeDetection/DHIF/Train/HSI/patch-%dth.mat'%(file_id)
            val_path_rgb = '/home2/szx/Dataset/TreeDetection/DHIF/Train/RGB/patch-%dth.mat'%(file_id)
            print("["+val_path_hsi+"] starts val")
            data = hdf5.loadmat(val_path_hsi)

            HR_HSI[:, :, :, 0] = data['hsi']

            data = hdf5.loadmat(val_path_rgb)
            HR_MSI[:, :, :, 0] = data['rgb']
        # return HR_HSI[:,:2560:,:], HR_MSI[:,:2560:,:]
            return HR_HSI[:, :, :, :], HR_MSI[:, :, :, :]
        # test_HR_HSI = torch.from_numpy(test_HR_HSI)
        # test_HR_HSI_max = torch.max(test_HR_HSI)
        # test_HR_HSI_min = torch.min(test_HR_HSI)
        # test_HR_HSI = (test_HR_HSI - test_HR_HSI_min) / (test_HR_HSI_max - test_HR_HSI_min)
        #
        # test_HR_MSI = torch.from_numpy(test_HR_MSI)
        # test_HR_MSI_max = torch.max(test_HR_MSI)
        # test_HR_MSI_min = torch.min(test_HR_MSI)
        # test_HR_MSI = (test_HR_MSI - test_HR_MSI_min) / (test_HR_MSI_max - test_HR_MSI_min)
        #
        # test_HR_HSI = test_HR_HSI.numpy()
        # test_HR_MSI = test_HR_MSI.numpy()

        test_HR_HSI, test_HR_MSI = val_prepare_data()
        dataset = cave_dataset(opt, test_HR_HSI, test_HR_MSI, istrain=False)
        loader_train = tud.DataLoader(dataset, batch_size=1)
        model = model.eval()
        model = dataparallel(model, 1)
        psnr_total = 0
        k = 0
        for j, (LR, RGB, HR) in enumerate(loader_train):
            with torch.no_grad():
                out = model(RGB.cuda(), LR.cuda())
                result = out
                result = result.clamp(min=0., max=1.)
            psnr = compare_psnr(result.cpu().detach().numpy(), HR.numpy(), data_range=1.0)
            psnr_total = psnr_total + psnr
            k = k + 1
            print(psnr)
            #
            res = result.cpu().permute(2, 3, 1, 0).squeeze(3).numpy()
            # save_path = './Result/ssr/' + str(j + 1) + '.mat'
            # sio.savemat(save_path, {'res': res})

        print(k)
        print("Avg PSNR = %.4f" % (psnr_total / k))
