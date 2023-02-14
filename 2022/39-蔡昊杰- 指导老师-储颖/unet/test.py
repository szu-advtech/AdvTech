from train import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from data import *


if __name__ == "__main__":
    print(os.getcwd())

    net = Unet(3, 2)
    device = torch.device(
        "cuda:0"if torch.cuda.is_available() else "cpu")  # 检测是否有GPU加速
    net.to(device)  # 网络放入GPU里加速
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('seccessful load weights')
    else:
        print('not successful load weights')

    testset = PVDataset(test_img_path, test_label_path)
    test_loader = DataLoader(testset, 1, shuffle=False)

    loss_list = []
    loss_sum = 0
    loss_func = nn.BCEWithLogitsLoss()

    # 测试一张图片的显示结果
    #test_img = readTif(test_img_path + "/" + '00010.tif')
    #test_label = readTif(test_label_path + "/" + '00010.tif')
    for i, (test_img, test_label) in enumerate(test_loader):
        test_img = test_img.to(device)
        test_label = test_label.to(device)
        

        out = net(test_img)
        test_label = test_label[:, :2, :, :, ]
        train_loss = loss_func(out, test_label)
        loss_sum += train_loss
        print(f'0-{i}-test_loss===>>{train_loss.item()}')
        loss_list.append(train_loss.item())
        print(out.shape)

        _test_label = test_label[0]
        _out = out[0]
        _test_label = _test_label[:1, :, :, ]
        _out = _out[:1, :, :, ]
        _test_label =_test_label.expand(3, 200, 200)      # 将维度拓展 (1, 200, 200)->(3, 200, 200),只有该维度的数量为1才可以expand
        _out = _out.expand(3, 200, 200)  

        img = torch.stack([test_img[0], _test_label, _out], dim=0)   
        save_image(out, f'{test_out_path}/{i}.tif')
        save_image(test_label, f'{test_label_path2}/{i}.tif')
        save_image(img, f'{save_test_img_path}/{i}.tif')

    with open(test_loss_path, 'a') as f1:
        f1.write(str(loss_list) + '\n')     # 换行
    print(len(test_loader))
    print('average test loss == ', loss_sum / len(test_loader))
