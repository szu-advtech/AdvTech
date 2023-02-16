import datetime

import torch
import numpy as np
import random
import time
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch import nn, optim
from tqdm import tqdm
from Model import RHDN, TTSN, Baseline
from dataloader import DataSet
from torch.utils.data import DataLoader
from metrics import SAM, ERGAS, rmse, scc
import os
import spectral as spy


def build_model(name):
    if name == "RHDN":
        model = RHDN(79, 23, RHDB_num1=2, RHDB_num2=4)  # 构建模型
    elif name == "TTSN":
        model = TTSN(79, 23, TSB_num1=2, TSB_num2=4)  # 构建模型
    else:
        model = Baseline(102, RHDB_num=8)  # 构建模型
    return model


def RHDN_train(train, batch_size, epochs, learning_rate):
    model = build_model(model_name)
    model.to(device)
    # model.load_state_dict(torch.load(weights_path + '/new_weights.pth', map_location=device))  # 预训练模型

    optim_model = optim.Adam(model.parameters(), lr=learning_rate)  # 优化器
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_model, milestones=[150, 300, 450], gamma=0.1, last_epoch=-1)
    best_loss = 1
    best_epoch = -1
    loss_fn_l1 = nn.L1Loss()

    # 数据加载
    train_db = DataSet(data_path, flag='train')
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    test_db = DataSet(data_path, flag='test')
    test_loader = DataLoader(test_db, batch_size=1, num_workers=0)

    if train:

        with SummaryWriter(logs_path) as writer:
            for epoch in range(epochs):
                model.train()
                losses = 0
                step = 0
                for (LRHS, PAN, gtHS) in tqdm(train_loader, total=len(train_loader), desc=f"[epoch {epoch}] "
                                                                                          f"training data..."):
                    step += 1

                    LRHS = F.interpolate(LRHS, mode='bilinear', scale_factor=4)  # 双线性插值上采样
                    LRHS = LRHS.type(torch.float).to(device)
                    PAN = PAN.type(torch.float).to(device)
                    gtHS = gtHS.type(torch.float).to(device)

                    optim_model.zero_grad()
                    HS_fusion = model(LRHS, PAN)
                    loss = loss_fn_l1(HS_fusion, gtHS)
                    losses = losses + loss.detach().cpu().item()
                    loss.backward()
                    optim_model.step()

                scheduler.step()

                model.eval()

                with torch.no_grad():
                    eval_losses = []
                    for (LRHS, PAN, gtHS) in tqdm(test_loader, total=len(test_loader), desc=f"testing data..."):
                        LRHS = F.interpolate(LRHS, mode='bilinear', scale_factor=4)
                        LRHS = LRHS.type(torch.float).to(device)
                        PAN = PAN.type(torch.float).to(device)
                        gtHS = gtHS.type(torch.float).to(device)

                        HS_fusion = model(LRHS, PAN)

                        eval_loss = loss_fn_l1(HS_fusion, gtHS)
                        eval_losses.append(eval_loss.detach().cpu().item())
                eval_loss = sum(eval_losses) / len(eval_losses)

                if best_loss > eval_loss:  # 保存最优模型参数
                    best_loss = eval_loss
                    best_epoch = epoch
                    if not os.path.exists(weights_path):
                        os.makedirs(weights_path)
                    torch.save(model.state_dict(), weights_path + '/best_weights.pth')

                if not os.path.exists(weights_path):  # 保存最新模型参数
                    os.makedirs(weights_path)
                torch.save(model.state_dict(), weights_path + '/new_weights.pth')
                if (epoch + 1) % 50 == 0:  # 50轮保存一次
                    torch.save(model.state_dict(), weights_path + '/epoch{}.pth'.format(epoch))
                print("The train loss of epoch {} :{}---".format(epoch,
                                                                 round(losses / step, 4)) + datetime.datetime.strftime(
                    datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
                writer.add_scalar("train_loss", round(losses / step, 4), epoch)

                print("The eval loss of epoch {} :{}---".format(epoch,
                                                                round(eval_loss, 4)) + datetime.datetime.strftime(
                    datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
                writer.add_scalar("eval_loss", round(eval_loss, 4), epoch)

        print("best_loss:", best_loss, "best_epoch:", best_epoch)

    else:

        # 模型加载
        model.load_state_dict(torch.load(weights_path + '/best_weights.pth', map_location=device))

        model.eval()
        index_test = 1
        with torch.no_grad():
            ccs = []
            sams = []
            rmses = []
            ergases = []
            for (LRHS, PAN, gtHS) in tqdm(test_loader, total=len(test_loader), desc=f"testing data..."):
                LRHS_copy = LRHS
                LRHS = F.interpolate(LRHS, mode='bilinear', scale_factor=4)
                LRHS = LRHS.type(torch.float).to(device)
                PAN = PAN.type(torch.float).to(device)
                gtHS = gtHS.type(torch.float).to(device)

                HS_fusion = model(LRHS, PAN)

                HS_fusion = np.array(HS_fusion.cpu()).squeeze().transpose(1, 2, 0)  # 去掉大小1的维度   CHW->HWC  放CPU上
                gtHS = np.array(gtHS.cpu()).squeeze().transpose(1, 2, 0)
                LRHS = np.array(LRHS_copy.cpu()).squeeze().transpose(1, 2, 0)

                path = os.path.join(fusion_path, model_name + '_' + str(index_test) + '.mat')
                result_path = os.path.join(results_path, model_name + '_' + str(index_test) + '.jpg')
                sio.savemat(path, {'b': HS_fusion.transpose(2, 0, 1)})

                spy.save_rgb(result_path, HS_fusion, rgb[set_name])

                # 评价指标
                ccs.append(scc(HS_fusion, gtHS))  # 交叉相关CC
                sams.append(SAM(HS_fusion, gtHS))
                rmses.append(rmse(HS_fusion, gtHS) / gtHS.shape[2])
                ergases.append(ERGAS(HS_fusion, gtHS, LRHS))

                index_test = index_test + 1
            print("CC:", sum(ccs) / len(ccs))
            print("SAM:", sum(sams) / len(sams))
            print("RMSE:", sum(rmses) / len(rmses))
            print("ERGAS:", sum(ergases) / len(ergases))
            print('test finished!')


if __name__ == '__main__':
    start_time = time.time()
    print("********************Start Running********************")
    # 设置随机种子，方便复现实验
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rgb = {"pavia": [55, 30, 5]}
    set_name = "pavia"

    # 所使用的模型名称
    # model_name = "RHDN"
    model_name = "TTSN"
    # model_name = "Baseline"

    # 数据加载路径
    data_path = os.path.abspath('../data/' + set_name).replace('\\', '/')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # 权重参数保存路径
    weights_path = os.path.abspath('../weights/' + set_name + '/' + model_name).replace('\\', '/')
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    # 融合结果mat保存路径
    fusion_path = os.path.abspath('../fusion/' + set_name + '/' + model_name).replace('\\', '/')
    if not os.path.exists(fusion_path):
        os.makedirs(fusion_path)
    # 日志保存路径
    logs_path = os.path.abspath("../logs/" + set_name + "/" + model_name).replace('\\', '/')
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    # 融合结果图片保存路径
    results_path = os.path.abspath('../results/' + set_name + '/' + model_name).replace('\\', '/')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    RHDN_train(train=False, batch_size=1, epochs=500, learning_rate=0.001)

    end_time = time.time()
    print("Running is Over!\nRunning time is {}s".format(end_time - start_time))
