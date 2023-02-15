import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4, 5, 6, 7"
import logging
import torch
from torch import nn
from model.Fcn2 import Fcn2
from dataset.dataset import M_v_dataset
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
import time


def set_logger():
    save_dir = r"/home/kaiyi/camera_view_pretrain/CameraViewPretrain/ckp2"
    sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(save_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger2 = logging.getLogger()
    logger2.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s",
                                     "%m-%d %H:%M:%S")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger2.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(os.path.join(save_dir, 'train.log'))
    fileHandler.setFormatter(logFormatter)
    logger2.addHandler(fileHandler)
    return logger2, save_dir


EPOCH = 500
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
device = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']


train_dataset = M_v_dataset(view=2)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=True)

net2 = Fcn2().cuda().to(device[1])
optimizer2 = optim.SGD(net2.parameters(), momentum=0.9, lr=LEARNING_RATE, weight_decay=1e-4)
mse_loss = nn.MSELoss()


def train(epoch, logger2, save_dir):
    mae = 0.
    mse = 0.
    count = 0.
    total_loss = 0.
    net2.train()
    start_time = time.time()
    for step, (img2, label2) in enumerate(train_dataloader):
        img2 = img2.cuda().to(device[1])
        label2 = label2.cuda().to(device[1])
        with torch.autograd.set_grad_enabled(True):
            loss2 = torch.tensor(0).float().cuda().to(device[1])
            out2 = net2(img2)

            for i in range(BATCH_SIZE):
                loss2 += mse_loss(out2[i].reshape(-1, 1), label2[i].reshape(-1, 1)) / BATCH_SIZE
                pre = torch.sum(out2[i]) / 1000
                gt = torch.sum(label2[i]) / 1000
                res = pre - gt
                print("pre: {},     gt: {},     res: {},     loss: {}".format(pre.data, gt.data, res.data, loss2.data))

                mae += torch.abs(res)
                mse += res * res
                count += 1
                #print(count)

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            total_loss += loss2

    logger2.info("loss: {}, mse: {}, mae {}, time: {}".format(total_loss / count, torch.sqrt(mse / count),
                                                              mae / count, time.time() - start_time))
    model_state_dic = net2.state_dict()
    save_path = os.path.join(save_dir, '{}_ckpt.tar'.format(epoch))
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer2.state_dict(),
        'model_state_dict': model_state_dic,
    }, save_path)


def main():
    logger2, save_dir = set_logger()
    for e in range(EPOCH):
        logger2.info("-------epoch {}-------".format(e))
        train(e, logger2, save_dir)


if __name__ == '__main__':
    main()



