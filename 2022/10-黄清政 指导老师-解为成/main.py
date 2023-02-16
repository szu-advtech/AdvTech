import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import dataset.get_raf as dataset
from loss.loss_All import vector_triplet_loss,matrix_triplet_loss
from utils import Bar, Logger, AverageMeter, accuracy
from model.EPMG_model import EPMG
from model.EFE_model import EFE
from model.EMR_model import EMR
from torchsummary import summary


parser = argparse.ArgumentParser(description='PyTorch task1 Training')

# Optimization options
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--batch-size', default=210, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--first-step-lr', '--learning-rate1', default=1e-3, type=float,
                    metavar='LR', help='initial first step learning rate')
parser.add_argument('--second-step-lr', '--learning-rate2', default=1e-4, type=float,
                    metavar='LR', help='initial first step learning rate')
parser.add_argument('--third-step-lr', '--learning-rate3', default=1e-3, type=float,
                    metavar='LR', help='initial first step learning rate')

parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
# Miscs
parser.add_argument('--manualSeed', type=int, default=5, help='manual seed')

#Device options
parser.add_argument('--gpu', default='5,6', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--train-iteration', type=int, default=60,
                        help='Number of iteration per epoch')
# Checkpoints
parser.add_argument('--out', default='/data/qzhuang/Projects/Task1/result/',
                        help='Directory to output the result')
# Dataload
parser.add_argument('--train-root', type=str, default="/data/qzhuang/datasets/RAF/train/",
                        help="root path to train data directory")
parser.add_argument('--test-root', type=str, default="/data/qzhuang/datasets/RAF/test/",
                        help="root path to test data directory")
parser.add_argument('--label-train', default="/data/qzhuang/datasets/RAF/Annotations/train.txt", type=str, help='')
parser.add_argument('--label-test', default="/data/qzhuang/datasets/RAF/Annotations/test.txt", type=str, help='')

parser.add_argument('--EFE_GT', '--EFE_gradient_threshold', default=1.6, type=float,
                    metavar='GT', help='EFE_net gradient threshold')
parser.add_argument('--EMR_GT', '--EMR_gradient_threshold', default=1.6, type=float,
                    metavar='GT', help='EFE_net gradient threshold')
parser.add_argument('--EPMG_GT', '--EPMG_gradient_threshold', default=1.6, type=float,
                    metavar='GT', help='EFE_net gradient threshold')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
# 设置随机种子，使得每次产生的随机数都一样
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)


best_acc1 = 0  # best test accuracy
best_acc2 = 0
Pretrain = False
def main():
    global best_acc1,best_acc2
    global Pretrain

    # Gets the address of the pre-training parameter
    Pretrain_par_path = []
    Pretrain_par_path.append("/data/hqz/Projects/Task1/result/FC_Pre/test/EFE_net_epoch_34_params.pth")
    Pretrain_par_path.append("/data/hqz/Projects/Task1/result/FC_Pre/test/EMR_net_epoch_34_params.pth")
    Pretrain_par_path.append("/data/hqz/Projects/Task1/result/FC_Pre/test/EPMG_net_epoch_34_params.pth")

    # Data
    print(f'==> Preparing RAF-DB')
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    gray = transforms.Grayscale()
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])
    # load data
    train_set, triplet_set_train,  test_set = dataset.get_raf(args.train_root, args.label_train, args.test_root, args.label_test,transform_train=transform_train, transform_val=transform_val)

    # data ,label
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=args.num_workers, drop_last=True)
    triplet_set_loader = data.DataLoader(triplet_set_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.num_workers,drop_last=True)

    # Model
    print("==> creating model")
    def create_model():
        EFE_net = EFE()
        EMR_net = EMR()
        EPMG_net = EPMG()
        # Train with multiple Gpus
        EFE_net = torch.nn.DataParallel(EFE_net).cuda()
        EMR_net = torch.nn.DataParallel(EMR_net).cuda()
        EPMG_net = torch.nn.DataParallel(EPMG_net).cuda()
        return EFE_net,EMR_net,EPMG_net

    EFE_net, EMR_net, EPMG_net = create_model()

    if Pretrain:
        print('Load the pre-training parameters')
        EFE_net.load_state_dict(torch.load(Pretrain_par_path[0]))
        EMR_net.load_state_dict(torch.load(Pretrain_par_path[1]))
        EPMG_net.load_state_dict(torch.load(Pretrain_par_path[2]))

    # Benchmark Train with multiple Gpus
    cudnn.benchmark = True

    print('    Total params: %.2fM' % ((sum(p.numel() for p in EFE_net.parameters())/1000000.0) +
                                       (sum(p.numel() for p in EMR_net.parameters())/1000000.0) +
                                       (sum(p.numel() for p in EPMG_net.parameters())/1000000.0)))

    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    # Loss of triplet
    criterion_triplet_vector = vector_triplet_loss
    criterion_triplet_matrix = matrix_triplet_loss
    # Mean square loss
    mseloss = nn.MSELoss(reduction='none')
    # optimizer
    optimizer_first = optim.SGD(EFE_net.parameters(), lr=args.first_step_lr, momentum=0.9)
    optimizer_second = optim.SGD([ {'params': EFE_net.parameters()},{'params': EMR_net.parameters()} ], lr=args.second_step_lr, momentum=0.9)
    optimizer_third = optim.SGD([ {'params': EFE_net.parameters()}, {'params': EPMG_net.parameters()} ], lr=args.third_step_lr, momentum=0.9)

    logger = Logger(os.path.join(args.out, 'log.txt'), title='RAF')
    logger.set_names(['Train Loss EFE', 'Train Loss EMR', 'Train Loss EPMG','Test Loss1','Test Loss2','Test_Acc1','Test Acc2.'])

    test_accs = []
    start_epoch = 1
    # Train and val
    for epoch in range(start_epoch, args.epochs + 1):

        train_loss_EFE, train_loss_EMR, train_loss_EPMG = train(train_loader,triplet_set_loader,EFE_net,EMR_net,EPMG_net, criterion_ce,mseloss,criterion_triplet_vector,criterion_triplet_matrix,optimizer_first,optimizer_second,optimizer_third, epoch,use_cuda,gray)

        train_loss1, train_loss2,train_acc1,train_acc2= validate(train_loader, EFE_net, EPMG_net,criterion_ce, use_cuda, mode='train Stats')
        test_loss1,test_loss2 ,test_acc1,test_acc2 = validate(test_loader, EFE_net,EPMG_net, criterion_ce, use_cuda, mode='Test Stats')

        # append logger file
        logger.append([train_loss_EFE, train_loss_EMR, train_loss_EPMG, test_loss1, test_loss2, test_acc1,test_acc2])
        print('train_loss_EFE:{},train_loss_EMR:{},train_loss_EPMG:{}'.format(train_loss_EFE,train_loss_EMR,train_loss_EPMG))

        # save model
        is_best = (test_acc1 > best_acc1)
        best_acc1 = max(test_acc1, best_acc1)
        best_acc2 = max(test_acc2, best_acc2)

        if  is_best:
            torch.save(EFE_net.state_dict(), args.out + '/best_EFE_net_epoch_'+str(epoch) +'_params.pth')
            torch.save(EMR_net.state_dict(), args.out + '/best_EMR_net_epoch_' + str(epoch) + '_params.pth')
            torch.save(EPMG_net.state_dict(), args.out + '/best_EPMG_net_epoch_' + str(epoch) + '_params.pth')

        test_accs.append([test_acc1,test_acc2])
    logger.close()
    print('Best acc1:{},Best acc2:{}'.format(best_acc1,best_acc2))
    test_accs = np.array(test_accs)
    np.save("/data/qzhuang/Projects/Task1/result/test_acc/test_accs.npy", test_accs)  # 保存为.npy格式
def train(train_loader,triplet_set_loader,EFE_model,EMR_model,EPMG_model, criterion_ce,mseloss,criterion_triplet_vector, criterion_triplet_matrix,optimizer_first,optimizer_second,optimizer_third, epoch,use_cuda,gray):

    print('train epoch: %d' % epoch)
    loss_EFE = 0
    loss_EMR = 0
    loss_EPMG = 0

    # first step
    EFE_model.train()
    # Unseal the last frozen model
    for name, param in EFE_model.named_parameters():
        param.requires_grad = True
    index = 0
    for batch in train_loader:
        inputs, targets = batch
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        optimizer_first.zero_grad()
        _, _, pre_EFE = EFE_model(inputs)

        L_EFE = criterion_ce(pre_EFE, targets).mean()
        index += 1
        loss_EFE += L_EFE.item()
        L_EFE.backward()
        # against gradient explosion
        torch.nn.utils.clip_grad_norm(EFE_model.parameters(), args.EFE_GT)
        optimizer_first.step()

    # The average sample loss of one iteration over the entire training data set
    loss_EFE = loss_EFE / index


    # Loading triples
    triplet_set_iter = iter(triplet_set_loader)
    # second step
    EMR_model.train()
    # Freeze the first 5 layers of the VGG
    five_layers = 0
    for name, param in EFE_model.named_parameters():
        if five_layers < 10:
            five_layers += 1
            param.requires_grad = False
    index = 0
    # random sampling
    for batch_idx in range(args.train_iteration):
        try:
            inputs, targets = triplet_set_iter.next()
        except:
            triplet_set_iter = iter(triplet_set_loader)
            inputs, targets = triplet_set_iter.next()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        optimizer_second.zero_grad()
        mask, _ ,pre_EFE = EFE_model(inputs)
        Hn, output = EMR_model(mask)
        L_EFE = criterion_ce(pre_EFE,targets).mean()
        L_h = criterion_triplet_vector(Hn,targets)
        # change mask shape
        mask = mask.view(mask.shape[0],7,7)
        L_mask = mseloss(mask,output).sum(dim=(1,2)).mean()
        L_EMR = 2 * L_EFE + 100 * L_mask + 100 * L_h
        loss_EMR += L_EMR.item()
        index += 1
        L_EMR.backward()
        torch.nn.utils.clip_grad_norm(EFE_model.parameters(), args.EFE_GT)
        torch.nn.utils.clip_grad_norm(EMR_model.parameters(), args.EMR_GT)
        optimizer_second.step()
    loss_EMR = loss_EMR / index

    #third step
    EPMG_model.train()
    # Freeze the EFE module
    for name, param in EFE_model.named_parameters():
            param.requires_grad = False
    index = 0
    for batch_idx in range(args.train_iteration):
        try:
            inputs, targets = triplet_set_iter.next()
        except:
            triplet_set_iter = iter(triplet_set_loader)
            inputs, targets = triplet_set_iter.next()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        optimizer_third.zero_grad()
        _, Gmask,_ = EFE_model(inputs)
        e, c_e, Pic = EPMG_model(Gmask)
        L_embed = criterion_triplet_vector(e,targets)
        L_classify = criterion_ce(c_e, targets).mean()
        L_encoder = L_embed + L_classify
        Pic = Pic.view(Pic.shape[0], 224, 224)
        L_sim = criterion_triplet_matrix(Pic,targets)
        I = gray(inputs)
        I = I.view(I.shape[0],224,224)
        L_pattern = mseloss(Pic,I).sum(dim=(1,2))
        L_pattern = L_pattern.mean()
        L_decoder = 60 * L_pattern + 40 * L_sim
        L_EPMG = L_encoder + L_decoder
        loss_EPMG += L_EPMG.item()
        index += 1
        L_EPMG.backward()
        torch.nn.utils.clip_grad_norm(EPMG_model.parameters(), args.EPMG_GT)
        optimizer_third.step()
    loss_EPMG = loss_EPMG / index
    return  loss_EFE ,loss_EMR ,loss_EPMG


def validate(valloader, model1,model2, criterion, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to evaluate mode
    model1.eval()
    model2.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    losses1 = 0.
    losses2 = 0.
    prec1_nums = 0.
    prec2_nums = 0.

    with torch.no_grad():
        # 索引，数据
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output

            _, Gmask, pre_EFE = model1(inputs)
            # P可以进行可视化输出
            _,c_e,P = model2(Gmask)

            loss1 = criterion(pre_EFE, targets).mean()
            loss2 = criterion(c_e,targets).mean()

            # measure accuracy and record loss
            prec1_nums += accuracy(pre_EFE, targets).item()
            prec2_nums += accuracy(c_e,targets).item()

            top1 = prec1_nums / (float(targets.shape[0]) * (batch_idx + 1))
            top2 = prec2_nums / (float(targets.shape[0]) * (batch_idx + 1))

            losses1 += float(loss1.item())
            losses2 += float(loss2.item())

            losses1_avg = losses1 / (batch_idx + 1)
            losses2_avg = losses2 / (batch_idx + 1)
            # 在变化不明显
            # print(losses2_avg)

            # prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            # losses1.update(loss1.item(), inputs.size(0))
            # losses2.update(loss2.item(), inputs.size(0))
            # top1.update(prec1.item(), inputs.size(0))
            # top2.update(prec2.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Total: {total:} | Loss1: {loss1:.4f} |Loss2: {loss2:.4f} | Accuracy1: {top1: .4f}|Accuracy2: {top2: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        total=bar.elapsed_td,
                        loss1=losses1_avg,
                        loss2=losses2_avg,
                        top1=top1,
                        top2=top2,
                        )
            bar.next()
        bar.finish()
    return (losses1_avg,losses2_avg,top1, top2)



if __name__ == '__main__':
    main()
