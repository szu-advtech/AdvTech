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

from data_load import My1x1Dataset
from fuxian.model import BiLSTM
from transfo_model import ABDTransformer
import logging

logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='fuxian.log',
                    filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志, a是追加.模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'  # 日志格式
                    )

torch.manual_seed(0)
save_model_path = 'save/'


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_len = len(train_loader)
    for idx, data in enumerate(train_loader):
        x = (data[0]).to(device).to(torch.float32)
        tgt = (data[1]).to(device).to(torch.float32)
        pred = model(x)
        loss = F.mse_loss(pred, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % args.log_interval == 0:
            logging.info("Train Time:{}, epoch: {}, step: {} / {}, loss: {}".format(time.strftime("%Y-%m-%d%H:%M:%S"),
                                                                             epoch + 1, idx, train_len, loss.item()))


def evaluate(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    test_len = len(test_loader)
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            x = (data[0]).to(device).to(torch.float32)
            tgt = (data[1]).to(device).to(torch.float32)
            pred = model(x)
            loss = F.mse_loss(pred, tgt)
            test_loss += loss.item()

    test_loss /= len(test_loader)

    logging.info("Test Time:{}, epoch:{}, loss: {}".format(
        time.strftime("%Y-%m-%d%H:%M:%S"),
        epoch,
        test_loss))


def main():
    # 设置argparse
    parser = argparse.ArgumentParser(description="TRAINING")
    parser.add_argument('--device_ids', type=str, default='0,1,2,3', help="Training Devices")
    parser.add_argument('--epochs', type=int, default=100, help="Training Epoch")
    parser.add_argument('--log_interval', type=int, default=10, help="Log Interval")
    parser.add_argument('--local_rank', type=int, default=-1, help="DDP parameter, do not modify")
    parser.add_argument('--bs', type=int, default=256, help="DDP parameter, do not modify")

    args = parser.parse_args()
    # 设置持久化路径
    # resume = ''
    resume = save_model_path + f'/bilstm-1-train.pt'
    # 设置并行
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    # 加载模型
    model = BiLSTM().to(device)
    model.train()
    # 加载训练集和测试集
    dataset_train = My1x1Dataset(start_year=2012, end_year=2015, attr='temp')
    dataset_test = My1x1Dataset(start_year=2016, end_year=2016, attr='temp')
    train_loader = DataLoader(dataset_train, batch_size=args.bs, num_workers=32)
    test_loader = DataLoader(dataset_test, batch_size=args.bs, shuffle=False, num_workers=32)
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    old_epoch = 0
    # 持久化加载
    if resume != "":
        #  加载之前训过的模型的参数文件
        print(f"loading from {resume}")
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        old_epoch = checkpoint['epoch']
    # 开始训练
    for epoch in range(args.epochs):
        epoch += old_epoch
        save_path = save_model_path + f'/bilstm-{epoch}-train.pt'
        train(args, model, device, train_loader, optimizer, epoch)
        evaluate(args, model, device, test_loader, epoch)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)
        logging.info(f'saving: [{save_path}]')


if __name__ == '__main__':
    main()
