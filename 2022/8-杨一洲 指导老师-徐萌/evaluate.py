import torch.utils.data
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from data_load import My1x1Dataset
from fuxian.model import BiLSTM
from transfo_model import ABDTransformer
import logging
import sklearn.metrics  # 模型性能评价模块
import skimage.measure
from skimage.metrics import mean_squared_error as compare_mse

depth_list = open('depth.txt').readlines()
depth_list = list(map(lambda x: int(x), depth_list))

device = 'cuda:1'
epoch = 99
resume = f'save/bilstm-{epoch}-train.pt'
bs = 32
logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='evaluate_new.log',
                    filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志, a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'  # 日志格式
                    )


def evaluate(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    all_pred_y = None
    all_valid_y = None
    test_len = len(test_loader)
    idx = -1
    with torch.no_grad():
        for data in test_loader:
            idx += 1
            print(100 * idx / test_len, '%')
            x = (data[0]).to(device).to(torch.float32)
            tgt = (data[1]).to(device).to(torch.float32)
            pred = model(x)

            if all_valid_y is None:
                all_valid_y = tgt
            else:
                all_valid_y = torch.cat([all_valid_y, tgt], dim=0)
            if all_pred_y is None:
                all_pred_y = pred
            else:
                all_pred_y = torch.cat([all_pred_y, pred], dim=0)

    all_pred_y = torch.tensor(all_pred_y)
    all_valid_y = torch.tensor(all_valid_y)

    torch.save(all_pred_y, 'all_pred_y-m1')
    torch.save(all_valid_y, 'all_valid_y-m1')

    res = []
    for i in range(len(depth_list)):
        if int(depth_list[i]) > 1000:
            break

        pred_y = all_pred_y[:, i].cpu().numpy()
        valid_y = all_valid_y[:, i].cpu().numpy()

        # nan_idx = torch.where(torch.isnan(pred_y))[0].numpy()
        # pred_y = np.delete(pred_y, nan_idx, axis=0)
        # valid_y = np.delete(valid_y, nan_idx, axis=0)

        r2_score = sklearn.metrics.r2_score(valid_y, pred_y)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(valid_y, pred_y))
        nrmse = compare_mse(valid_y, pred_y)

        print('depth:' + str(depth_list[i]))
        print('R2_score:', r2_score)  # r2_score
        print('NRMSE:', nrmse)
        res.append([r2_score, nrmse])

    torch.save(res, 'res.ls')


def main():
    print(f"loading from {resume}")
    checkpoint = torch.load(resume)

    weights = checkpoint['model_state_dict']
    model = BiLSTM().to(device)

    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    model.load_state_dict(weights_dict)
    dataset_test = My1x1Dataset(start_year=2016, end_year=2016, attr='temp')
    test_loader = DataLoader(dataset_test, batch_size=bs, shuffle=False, num_workers=32, drop_last=True)
    evaluate(None, model, device, test_loader, epoch)


if __name__ == '__main__':
    main()
