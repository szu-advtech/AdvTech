# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import math
import random
import shutil
import sys
import os
import time
from pathlib import Path
import traceback

import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder,ImageFolderT,ImageFolderRGB
from compressai.zoo import image_models
from compressai.models import Guided_compresser, Master_compresser  
from compressai.zoo import load_state_dict
import torch.distributed as dist
import gc 

torch.backends.cudnn.benchmark=True

# bpp+mse
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, q):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = [256, 512, 1024, 2048, 4096,8192,10240]
        self.q = q

    def forward(self, output, target):
        #print('quality',self.q)
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        #print('out bpp&mse: ',out['bpp_loss'],out['mse_loss'])
        out["loss"] = self.lmbda[self.q]* out["mse_loss"] + out["bpp_loss"]

        return out

# 添加，并更新平均值
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 获取模块属性
class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

# 配置auxiliary和(mse+bpp)损失的优化器,aux优化entropy_bottleneck
def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch_guided(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm,quaility,scaler
):
    model.train()
    device = next(model.parameters()).device
    use = 0
    losses = []
    for i, d in enumerate(train_dataloader):
        #print('d.shape',d.shape)
        start = time.time()
        d = d.to(device)
        
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # out_net = model(d)

        # out_criterion = criterion(out_net, d)
        # out_criterion["loss"].backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        # optimizer.step()

        # aux_loss = model.aux_loss()
        # aux_loss.backward()
        # aux_optimizer.step()

        # 混合精度训练  # 在guided的时候开启混合精度训练速度反而下降
        with autocast():   #前后开启autocast
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            
        scaler.scale(out_criterion["loss"]).backward()  #为了梯度放大
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm) 
        scaler.step(optimizer)

        with autocast():
            aux_loss = model.aux_loss()
        scaler.scale(aux_loss).backward()  #为了梯度放大
        scaler.unscale_(aux_optimizer)
        scaler.step(aux_optimizer)
        scaler.update()



        losses.append(out_criterion["loss"].item())
        end = time.time()
        use += end-start
        if i % 10 == 0:
            print(f'q:{quaility},iter/s:{use/10}')
            use = 0
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.5f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.5f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.5f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
        gc.collect()
    return sum(losses)/len(losses)

def train_one_epoch_master(
    model, model_G,criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm,quaility,scaler
):
    model.train()
    device = next(model.parameters()).device

    use = 0
    losses = []
    for i, (x,guided) in enumerate(train_dataloader):
        start = time.time()
        x = x.to(device)
        guided = guided.to(device)
       
        with torch.no_grad():
            hidden = model_G(guided)['hidden']

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # out_net = model(x,guided,hidden) 
        # out_criterion = criterion(out_net, x)
        # out_criterion["loss"].backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm) 
        # optimizer.step()

        # aux_loss = model.aux_loss()
        # aux_loss.backward()
        # aux_optimizer.step()


        # 开启混合精度训练
        with autocast():   #前后开启autocast
            out_net = model(x,guided,hidden) 
            out_criterion = criterion(out_net, x)

        scaler.scale(out_criterion["loss"]).backward()  #为了梯度放大
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)  # 似乎如果没有梯度裁剪，那么会在后面裁剪（时间不变）
        scaler.step(optimizer)

        with autocast():
            aux_loss = model.aux_loss()
        scaler.scale(aux_loss).backward()
        scaler.unscale_(aux_optimizer)
        scaler.step(aux_optimizer)
        scaler.update()

        
        end = time.time()

        use += end-start
        losses.append(out_criterion["loss"].item())
        if i % 10 == 0:
            use /=10
            print(f'q:{quaility},iter/s:{use}')
            use = 0
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(x)}/{len(train_dataloader.dataset)}"        # 涉及到batch-size的计算
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.5f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.5f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.5f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
        gc.collect()
    return sum(losses)/len(losses)

def save_checkpoint(state, is_best,prefix, filename="checkpoint.pth.tar"):
    # if not os.path.exists('model'):
    #     os.mkdir('model')
    # prefix = 'model/'+prefix
    torch.save(state, prefix+filename)
    if is_best:  # 模型备份，或者供验证集保存
        shutil.copyfile(prefix+filename, prefix+"checkpoint_best_loss.pth.tar")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        help="Model architecture (default: %(default)s)",
    )
    # 直接传入训练集的文件夹，不考虑验证集
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--quality", type=int,default=4,help="quality [0-5)"
    )
    parser.add_argument(
       "--channel", type=int,default=1,help="image channel"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument( 
        "--patch-size", # 这里的-也是能和_匹配的
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # cuda和cpu
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float,default=256,help="Set random seed for reproducibility"
    )
    
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    # Guided_comprresser_4_thermal_8_bit_x_checkpoint.pth.tar
    parser.add_argument("--checkpoint", type=str,default=None,nargs='*', help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    
    if args.model.find("Guided") != -1 :
        net = Guided_compresser(channel=args.channel)
        net = net.to('cuda')
        train_dataset = ImageFolderT(args.dataset, size=args.patch_size,channel = args.channel)  # 在RGBT论文复现中，patchsize固定
    else:
        master_chl = args.channel
        if master_chl == 3:
            guided_chl = 1
        else:
            guided_chl = 3
        net = Master_compresser(width=512,height=640,channel=master_chl)                     # 在RGBT论文复现中，wh固定
        net = net.to('cuda')
        netG = Guided_compresser(channel=guided_chl)
        netG = netG.to('cuda')
        train_dataset = ImageFolderRGB(args.dataset, size=args.patch_size,channel = master_chl)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",patience=3,factor=0.7)  # 主要依据数据集的大小来决定
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,40,gamma=0.5,last_epoch=-1,verbose=False)
    criterion = RateDistortionLoss(q=args.quality)

    last_epoch = 0
    if args.checkpoint:   
        print("Loading guided checkpoint", args.checkpoint)
        checkpoint1 = torch.load(args.checkpoint[0], map_location="cpu")
        last_epoch = checkpoint1['epoch']+1
        
        if args.model.find("Guided") != -1:                  # guided断点重训
            net.load_state_dict(load_state_dict(checkpoint1['state_dict']))
            optimizer.load_state_dict(checkpoint1['optimizer'])
            aux_optimizer.load_state_dict(checkpoint1['aux_optimizer'])
            lr_scheduler.load_state_dict(checkpoint1['lr_scheduler'])
           
        else:
            last_epoch = 0
            netG.load_state_dict(load_state_dict(checkpoint1['state_dict']))  # 仅仅是guided
            netG = CustomDataParallel(netG)
            if len(args.checkpoint) == 2:                    # master也断点重训
                checkpoint2 = torch.load(args.checkpoint[1], map_location="cpu")
                last_epoch = checkpoint2['epoch']+1
                net.load_state_dict(load_state_dict(checkpoint2['state_dict']))
                optimizer.load_state_dict(checkpoint2['optimizer'])
                aux_optimizer.load_state_dict(checkpoint2['aux_optimizer'])
                lr_scheduler.load_state_dict(checkpoint2['lr_scheduler'])
    
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    modal = 'rgb_' if args.channel == 3 else 'x_'
    checkpoint_prefix = f'{args.model}_{args.quality}_{Path(args.dataset).stem}_{modal}'
    

    scaler = GradScaler() #训练前实例化一个GradScaler对象
    min_loss = 0xfffffff
    try:
        for epoch in range(last_epoch,args.epochs):
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            if args.model.find("Guided") != -1 :
                loss = train_one_epoch_guided(
                                        net, 
                                        criterion,
                                        train_dataloader,
                                        optimizer,
                                        aux_optimizer,
                                        epoch,
                                        args.clip_max_norm,
                                        args.quality,
                                        scaler
                                        
                                    )
            else:
                loss = train_one_epoch_master(
                                        net,
                                        netG,
                                        criterion,
                                        train_dataloader,
                                        optimizer,
                                        aux_optimizer,
                                        epoch,
                                        args.clip_max_norm,
                                        args.quality,
                                        scaler
                                    )

            # lr_scheduler.step(loss)
            lr_scheduler.step()
            save_it = False
            if loss < min_loss:
                save_it = True
                min_loss = loss

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    save_it, # 默认是True
                    checkpoint_prefix
                )
    except Exception as e:
        # 输出到控制台
        traceback.print_exc()
        print('Exception happens at',time.asctime( time.localtime(time.time()) ))
        
        # 输出到文件中
        trace_file_path = 'error.log'
        trace_file=open(trace_file_path, 'a')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),checkpoint_prefix,file=trace_file) #记录报错的时间
        traceback.print_exc(file=trace_file) #输出报错到文件 
        trace_file.close()
    else:
        print('Task is finished.')    
    
if __name__ == "__main__":
    main(sys.argv[1:])




