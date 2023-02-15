import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4, 5, 6, 7"
import logging
import torch
from torch import nn
from model.Fcn3 import Fcn3
from dataset.dataset import M_v_dataset
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
import time


def set_logger():
    save_dir = r"/home/kaiyi/camera_view_pretrain/CameraViewPretrain/ckp3"
    sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(save_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger3 = logging.getLogger()
    logger3.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s",
                                     "%m-%d %H:%M:%S")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger3.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(os.path.join(save_dir, 'train.log'))
    fileHandler.setFormatter(logFormatter)
    logger3.addHandler(fileHandler)
    return logger3, save_dir


EPOCH = 500
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
device = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']


train_dataset = M_v_dataset(view=3)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)

net3 = Fcn3().cuda().to(device[2])
optimizer3 = optim.SGD(net3.parameters(), momentum=0.9, lr=LEARNING_RATE, weight_decay=1e-4)

mse_loss = nn.MSELoss()


def train(epoch, logger3, save_dir):
    mae = 0.
    mse = 0.
    count = 0.
    total_loss = 0.
    net3.train()
    start_time = time.time()
    for step, (img3, label3) in enumerate(train_dataloader):
        img3 = img3.cuda().to(device[2])
        label3 = label3.cuda().to(device[2])
        with torch.autograd.set_grad_enabled(True):
            loss3 = torch.tensor(0).float().cuda().to(device[2])
            out3 = net3(img3)
            for i in range(BATCH_SIZE):
                loss3 += mse_loss(out3[i].reshape(-1, 1), label3[i].reshape(-1, 1)) / BATCH_SIZE
                pre = torch.sum(out3[i]) / 1000
                gt = torch.sum(label3[i]) / 1000
                res = pre - gt
                print("pre: {},     gt: {},     res: {},     loss: {}".format(pre.data, gt.data, res.data, loss3.data))

                mae += torch.abs(res)
                mse += res * res
                count += 1
                #print(count)
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()
            total_loss += loss3

    print(count)
    logger3.info("loss: {}, mse: {}, mae {}, time: {}".format(total_loss / count, torch.sqrt(mse / count),
                                                              mae / count, time.time() - start_time))
    model_state_dic = net3.state_dict()
    save_path = os.path.join(save_dir, '{}_ckpt.tar'.format(epoch))
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer3.state_dict(),
        'model_state_dict': model_state_dic,
    }, save_path)


def main():
    logger3, save_dir = set_logger()
    for e in range(EPOCH):
        logger3.info("-------epoch {}-------".format(e))
        train(e, logger3, save_dir)


if __name__ == '__main__':
    main()



