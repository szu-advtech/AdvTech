"""Run training."""

import shutil
import time
import numpy as np
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision
from CAM import show_feature_map

# from dataset import CoviarDataSet
from dataset import CoviarDataSet
from model import Model
from train_options import parser
from transforms import GroupCenterCrop
from transforms import GroupScale
import os
# from timesformer.models.vit import TimeSformer

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
SAVE_FREQ = 40
PRINT_FREQ = 20
best_prec1 = 0
from tensorboardX import SummaryWriter

exp_name='MMI-10-5-bacth-4-lr-0.001-resnet50+not-LStM--abs'
print("exp_name=",exp_name)
exp_path="/data/jlzhang/Py_protect/2022-08-05/pytorch-coviar-master/exp/iframe/"
exp_path=os.path.join(exp_path,exp_name)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)


writer = SummaryWriter(exp_path)
# writer2 = SummaryWriter("/data/zjl/192-torch/pytorch-coviar-master/exp/mv/")
# writer3 = SummaryWriter("/data/zjl/192-torch/pytorch-coviar-master/exp/residual/")
# from SCN import Res18Feature
def main():
    global args
    global best_prec1
    args = parser.parse_args()

    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    if args.data_name == 'ucf101':
        num_class = 101
    elif args.data_name == 'hmdb51':
        num_class = 51
    elif args.data_name == 'mmi':
        num_class = 6
    else:
        raise ValueError('Unknown dataset ' + args.data_name)
    #
    if args.arch in ['ts']:
        model = TimeSformer(img_size=224, num_classes=num_class, num_frames=args.num_segments,
                            attention_type='divided_space_time',
                            pretrained_model="/data/jlzhang/.cache/torch/hub/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth",
                            representation=args.representation)
    # if args.arch in ['scn']:
    #     model=Res18Feature(pretrained=imagenet_pretrained, drop_rate=args.drop_rate)
    else:
        model = Model(num_class, args.num_segments, args.representation,
                      base_model=args.arch)

    # for name in model.state_dict():
    #     print(name)
    #
    #     print(model.state_dict()['base_model.conv1.weight'])

    print('Total params is : %.2f M' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    if hasattr(model, 'flops'):
        flops = model.flops()
        print(f"number of GFLOPs: {flops / 1e9}")
    print(model)

    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.train_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=model.get_augmentation(),
            is_train=True,
            accumulate=(args.no_accumulation),
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=torchvision.transforms.Compose([
                GroupScale(int(model.scale_size)),
                GroupCenterCrop(model.crop_size),
            ]),
            is_train=False,
            accumulate=(args.no_accumulation),
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # torch.cuda.set_device(1)

    # model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    model = torch.nn.DataParallel(model)
    model.to(device)
    cudnn.benchmark = True

    params = model.parameters()

    # model_dict = model.state_dict()

    for name, param in model.named_parameters():
        # print(name)
        name2=name.split('.')[1]
        if name2 == 'mobilenet':
            param.requires_grad = False

    # params_dict = dict(model.named_parameters())
    # params = []
    #
    # for key, value in params_dict.items():
    #     decay_mult = 0.0 if 'bias' in key else 1.0
    #
    #     if ('module.base_model.conv1' in key
    #             or 'module.base_model.bn1' in key
    #             or 'data_bn' in key) and args.representation in ['mv', 'residual']:
    #         lr_mult = 0.1
    #     elif '.fc.' in key:
    #         lr_mult = 1.0
    #     else:
    #         lr_mult = 0.01
    #
    #     params += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]


    # optimizer = torch.optim.Adam(
    #     params,
    #     weight_decay=args.weight_decay,
    #     eps=0.001)

    optimizer = torch.optim.SGD(params, args.lr,
                                weight_decay=5e-5)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    for epoch in range(args.epochs):
        # cur_lr = adjust_learning_rate(optimizer, epoch, args.lr_steps, args.lr_decay)
        cur_lr = args.lr

        train(train_loader, model, criterion, optimizer, epoch, cur_lr)

        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, epoch)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            # with open ("/data/jlzhang/Py_protect/2022-08-05/pytorch-coviar-master/checkopint/I_best_Accracy.txt",'a') as f:
            #     f.write(f'{epoch},{best_prec1}'+'\n')
            if is_best or epoch % SAVE_FREQ == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                    },
                    is_best,
                    filename='checkpoint.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, cur_lr):
    batch_time = AverageMeter()  ##一个batch_size平均训练时间
    data_time = AverageMeter()  ##读取一个batch_size的平均时间
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()

    for i, (input, target,den_input) in enumerate(train_loader):

        data_time.update(time.time() - end)

        input = input.cuda()

        den_input=den_input.cuda()

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)

        target_var = torch.autograd.Variable(target)

        den_input_var = torch.autograd.Variable(den_input)

        if args.arch in ['ts']:
            input_var = input_var.transpose(1, 2)

        output,MI_1,MI_2 = model(input_var,den_input_var)

        # ret = torch.mean(MI_1) - torch.log(torch.mean(torch.exp(MI_2)))
        # input = input.view((-1, 3, 224, 224))
        # show_feature_map(input,feature,16,1,5)
        # break

        # if args.arch not in ['ts']:
        #     output = output.view((-1, args.num_segments) + output.size()[1:])
        #     output = torch.mean(output, dim=1)

        # x1=output[:,2,:]
        # x2=x1*0.5
        # output[:,2,:]=x2

        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # losses.update(loss.data[0], input.size(0))
        losses.update(loss.item(), input.size(0))
        # top1.update(prec1[0], input.size(0))
        top1.update(prec1.item(), input.size(0))

        # top5.update(prec5[0], input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # for name, param in model.named_parameters():
        #     name2 = name.split('.')[1]
        #     if name2 == 'mobilenet':
        #         print("param==",param)
        #         break

        if i % PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5,
                lr=cur_lr)))

            writer.add_scalar('train_loss', scalar_value=losses.avg, global_step=epoch)


def validate(val_loader, model, criterion, epoch):
    print("val=====================")
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target,den_input) in enumerate(val_loader):

        input = input.cuda()
        den_inpu=den_input.cuda()
        target = target.cuda()
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            den_input_var = torch.autograd.Variable(den_input)
        with torch.no_grad():
            target_var = torch.autograd.Variable(target)
        if args.arch in ['ts']:
            input_var = input_var.transpose(1, 2)

        output,MI_1,MI_2 = model(input_var,den_input_var)

        # ret = torch.mean(MI_1) - torch.log(torch.mean(torch.exp(MI_2)))

        # if args.arch not in ['ts']:
        #     output = output.view((-1, args.num_segments) + output.size()[1:])
        #     output = torch.mean(output, dim=1)

        # x1=output[:,2,:]
        # x2=x1*0.5
        # output[:,2,:]=x2
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader),
                batch_time=batch_time,
                loss=losses,
                top1=top1,
                top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))
    # with open(
    #         "/data/jlzhang/Py_protect/2022-08-05/pytorch-coviar-master/checkopint/I_vl_Accracy.txt",
    #         'a') as f:
    #     f.write(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
    #        .format(top1=top1, top5=top5, loss=losses))+ '\n')

    writer.add_scalar('val_accuracy', top1.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best, filename):
    filename = '_'.join((args.model_prefix, args.representation.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.model_prefix, args.representation.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    wd = args.weight_decay
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr * param_group['lr_mult']
    #     param_group['weight_decay'] = wd * param_group['decay_mult']
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


if __name__ == '__main__':
    # global exp_name
    # print("exp_name==", exp_name)
    seed = random.randint(1, 10000)
    # seed=7712
    setup_seed(seed)
    print("seed==", seed)
    print("GPU::", os.environ["CUDA_VISIBLE_DEVICES"])
    main()
