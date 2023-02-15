import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime
import thop

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of MVSNet')
parser.add_argument('--mode', default='test', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--train_epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
parser.add_argument('--numdepth', type=int, default=256, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default='./checkpoints/search/C1_2_PD/model_000000.ckpt', help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume',default=True, action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

# parse arguments and check
args = parser.parse_args()
if args.testpath is None:
    args.testpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger for mode "train" and "testall"
if args.mode == "train" or "test":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 3, args.numdepth, args.interval_scale)
test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, 8, shuffle=False, num_workers=4, drop_last=False)

# model
pre_model = MVSNet(refine=False)
pre_model.cuda()

if args.mode in ["train", "test"]:
    pre_model = nn.DataParallel(pre_model)

# load parameters
start_epoch = 0
if args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    pre_model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))

# new model
c = []
total = 0
for m in pre_model.modules():
    if isinstance(m, nn.BatchNorm3d):
        if m.weight.data.shape[0] == 1:
            c.append(m.weight.data.abs().clone().cpu().tolist())

z = [8, 16, 16, 32, 32, 64, 64]
opt_num = 3
cfg = []
t1 = 0
t2 = 0
t3 = 0
x = []
y = []
for i in range(len(c)):
    x.append(c[i][0])
    t1 += 1
    if t1 == opt_num:
        y.append(x)
        x = []
        t1 = 0
        t2 += 1
    if t2 == z[t3]:
        cfg.append(y)
        y = []
        t2 = 0
        t3 += 1
model = MVSNet(refine=False, cfg=cfg)
model.cuda()
'''a = torch.randn((1, 5, 3, 512, 640)).cuda()
b = torch.randn((1, 5, 4, 4)).cuda()
c = torch.randn((1, 192)).cuda()
flops, params = thop.profile(model, inputs=(a,b,c,))
flops, params = thop.clever_format([flops, params], "%.3f")
print('Number of model parameters: {}'.format(params))
print('Number of model FLOPs: {}'.format(flops))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in pre_model.parameters()])))'''

if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)

model_loss = mvsnet_loss
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

if args.mode == 'train':
    id = [0, 0, 0, 0, 0, 0, 0]
    flag_B = False
    flag_C = False
    for name, layer in model.named_modules():
        if 'upsample' in name:
            for n, l in pre_model.named_modules():
                if name in n:
                    if isinstance(l, nn.ConvTranspose3d) and isinstance(layer, nn.ConvTranspose3d):
                        layer.weight.data = l.weight.data.clone()
                    elif isinstance(l, nn.BatchNorm3d) and isinstance(layer, nn.BatchNorm3d):
                        layer.weight.data = l.weight.data.clone()

        elif 'feature' in name:
            for n, l in pre_model.named_modules():
                if name in n:
                    if isinstance(l, nn.Conv2d) and isinstance(layer, nn.Conv2d):
                        layer.weight.data = l.weight.data.clone()
                    elif isinstance(l, nn.BatchNorm2d) and isinstance(layer, nn.BatchNorm2d):
                        layer.weight.data = l.weight.data.clone()

        else:
            for i in range(7):
                if 'layer' + str(i+1) in name and (isinstance(layer, nn.BatchNorm3d) or isinstance(layer, nn.Conv3d)):
                    op = np.argmax(cfg[i][id[i]])
                    for n, l in pre_model.named_modules():
                        if name[:41] in n and 'ops.' + str(op) in n and name[-4:] in n:
                            if isinstance(l, nn.Conv3d) and isinstance(layer, nn.Conv3d):
                                layer.weight.data = l.weight.data.clone()
                                flag_B = True
                                break
                            elif isinstance(l, nn.BatchNorm3d) and isinstance(layer, nn.BatchNorm3d):
                                layer.weight.data = l.weight.data.clone()
                                flag_C = True
                                break
                    if flag_B and flag_C:
                        id[i] += 1
                        flag_B = False
                        flag_C = False

#resume
if (args.mode == "train" and args.resume) or args.mode == "test":
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1

# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    # training
    for epoch_idx in range(start_epoch, args.train_epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary, search=False)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.train_epochs, batch_idx,
                                                            len(TrainImgLoader), loss, time.time() - start_time))

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        # testing
        '''avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary, search=False)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx,
                                            args.train_epochs, batch_idx, len(TestImgLoader), loss, time.time() - start_time))
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())'''

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def test():
    global_step = 0
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        global_step = batch_idx

        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True, search=False)
        save_scalars(logger, 'test', scalar_outputs, global_step)
        save_images(logger, 'test', image_outputs, global_step)
        avg_test_scalars.update(scalar_outputs)
        print('Iter {}/{}, test loss = {:.3f}, abs_depth_error = {:.3f}, time = {:3f}'.format(batch_idx,
                                 len(TestImgLoader), loss, scalar_outputs["abs_depth_error"], time.time() - start_time))
        del scalar_outputs, image_outputs
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
    print("final", avg_test_scalars.mean())


def train_sample(sample, detailed_summary=False, search=True):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], search=search, training=True)
    depth_est = outputs["depth"]

    loss = model_loss(depth_est, depth_gt, mask)
    loss.backward()
    optimizer.step()

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]}
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
        scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
        scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
        scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True, search=False):
    model.eval()
    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], search=search, training=False)
    depth_est = outputs["depth"]

    loss = model_loss(depth_est, depth_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]}
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "profile":
        profile()
