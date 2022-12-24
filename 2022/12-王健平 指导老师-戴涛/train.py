import os
import torch.backends.cudnn as cudnn
import torch
from models import edgeSR_CNN, FSRCNN, ESPCN, edgeSR_MAX, edgeSR_TM, edgeSR_TR, edgeSR_TR_ECBSR
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datasets import TrainDataset, EvalDataset
from torch.utils.data.dataloader import DataLoader
import copy
from utils import AverageMeter, calc_psnr
from tqdm import tqdm
import argparse
import torchvision.transforms.functional as TF


# 超分用训练
if __name__ == '__main__':
    # ###################################################################
    # ############################## 初始化 ##############################
    # ###################################################################
    parser = argparse.ArgumentParser()
    # ############################### 必填 ###############################
    """网络模型"""
    # FSRCNN、edgeSR_MAX、edgeSR_CNN
    parser.add_argument('--net', type=str, required=True)
    """训练文件 h5格式"""
    parser.add_argument('--train-file', type=str, required=True)
    """评估文件 h5格式"""
    parser.add_argument('--eval-file', type=str, required=True)
    """输出文件夹"""
    parser.add_argument('--outputs-dir', type=str, required=True)
    """gpu选择"""
    # 0到n-1
    parser.add_argument('--gpu-id', type=int, required=True)
    # ############################### 选填 ###############################
    """超参"""
    parser.add_argument('--weights-file', type=str)
    """放大倍数"""
    parser.add_argument('--scale', type=int, default=2)
    """学习率"""
    parser.add_argument('--lr', type=float, default=1e-3)
    """batch大小"""
    parser.add_argument('--batch-size', type=int, default=16)
    """循环次数"""
    parser.add_argument('--num-epochs', type=int, default=20)
    """工作进程"""
    # 调度数据进入内存 每个进程一个batch
    parser.add_argument('--num-workers', type=int, default=8)
    """随机数种子"""
    # 不修改保证每次运行都是相同结果
    parser.add_argument('--seed', type=int, default=123)
    """是否采用分组卷积"""
    parser.add_argument('--group-conv', type=int, default=1)
    args = parser.parse_args()
    # ##################################################################
    """文件夹初始化"""
    args.outputs_dir = os.path.join(args.outputs_dir, '{}_x{}'.format(args.net, args.scale))
    # 输出文件夹初始化
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    # 网络输入的数据维度或类型变化不大时，cuDNN的auto-tunner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    cudnn.benchmark = True
    """设备选择"""
    if torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')
    # 生成随机数种子 注意：是每次运行test.py时对应语句随机数相同，而不是每个随机数语句生成数一样
    torch.manual_seed(args.seed)
    """网络初始化"""
    # 网络模型、损失函数、优化器初始化
    # nn.MSELoss 默认参数reduction = 'mean'
    # ##############################    FSRCNN   #############################
    if args.net == 'FSRCNN':
        model = FSRCNN(args.scale).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam([
            {'params': model.first_part.parameters()},
            {'params': model.mid_part.parameters()},
            {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
        ], lr=args.lr)
    # ############################   edgeSR   ###############################
    # 一层卷积
    # ############################ edgeSR_MAX ###############################
    elif args.net == 'edgeSR_MAX':
        model = edgeSR_MAX(args.scale,groups=args.group_conv).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # ########################    edgeSR_TM    ##########################
    elif args.net == 'edgeSR_TM':
        model = edgeSR_TM(args.scale,groups=args.group_conv).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # ########################    edgeSR_TR    ##########################
    elif args.net == 'edgeSR_TR':
        model = edgeSR_TR(args.scale,groups=args.group_conv).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # ########################    edgeSR_TR_ECBSR    ##########################
    elif args.net == 'edgeSR_TR_ECBSR':
        model = edgeSR_TR_ECBSR(args.scale,groups=args.group_conv, flag=1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 多层卷积
    # ############################ edgeSR_CNN ###############################
    elif args.net == 'edgeSR_CNN':
        model = edgeSR_CNN(args.scale,groups=args.group_conv).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # ########################################################################

    # ##############################    ESPCN   #############################
    elif args.net == 'ESPCN':
        model = ESPCN(args.scale).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # ########################################################################

    """训练集"""
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # 锁页内存 内存中的tensor不交换到硬盘的虚拟内存 可能会导致爆内存
        pin_memory=True
    )
    """评估集/测试集"""
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=1
    )
    """记录训练最佳epoch及模型参数"""
    # 深复制模型状态字典
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0


    """训练"""
    last_loss, last_time = 0, 0
    for epoch in range(args.num_epochs):
        # 训练模式
        model.train()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(train_dataset)- len(train_dataset) % args.batch_size), ncols= 80) as t:
            # 动态进度条
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))
            for data in train_dataloader:
                # 读取输入图片和标签 放进gpu训练
                # inputs是缩小n倍后的图片
                # labels是正常图片也就是gt
                # 注意：这里的图片指的是从各个图片中切割出的patch
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                # 输出所有像素损失均值
                loss = criterion(preds, labels)
                # 均值*像素数得到总损失
                epoch_losses.update(loss.item(), len(inputs))
                # 优化器梯度清0
                optimizer.zero_grad()
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()
                # 动态进度条更新
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        # 保存每个epoch超参
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        # 评估模式
        model.eval()
        epoch_psnr = AverageMeter()
        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                # 取消梯度下降，预测后每个像素概率范围限定为0到1
                preds = model(inputs).clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        # 记录最佳epoch
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        #
        if (abs(last_loss) * 0.99) <= abs(loss):
            if epoch != 0:
                if last_time > 2:
                    break
                else:
                    last_time += 1
            else:
                last_loss = loss
        else:
            last_time = 0
            last_loss = loss

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))



































































