import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LinearLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from args.base_args import get_base_args
from dataset.coco_handler import CocoHandler, CocoDataset, get_seg
from models import Model
from models.losses import KLD_Loss, GAN_Loss, GAN_FLoss, VGG_Loss
from eval import eval
from sklearn.cluster import KMeans
    

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '64263'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train(rank, model, epoch, train_loader, sampler, vgg_losser, optimizer_g, optimizer_d):
    
    model.train()
    sampler.set_epoch(epoch)
    
    gan_losser = GAN_Loss("hinge")
    gan_flosser = GAN_FLoss()

    for batch_idx, data in enumerate(train_loader):
        if model.module.is_clip:
            (real_img, label, inst, feature) = data
            feature = feature.float().to(rank)
        else:
            (real_img, label, inst) = data
            feature = None
        real_img, label, inst = real_img.to(rank), label.to(rank), inst.to(rank)
        seg = get_seg(rank, label, inst)

        # Encoder
        if model.module.is_encode:
            mu, log_s = model.module.encode_img(real_img)
            z = model.module.get_nosie(mu, log_s)
        else:
            size = model.module.net_g.start_size
            z = torch.nn.functional.interpolate(seg, (size, size))
            # z = torch.randn([real_img.shape[0], 256]).to(rank)

        # fake_img = model.module.gen_img(z, seg)
        fake_img = model.module.gen_img(z, seg, feature)
        
        # Det
        real_score = model.module.det_img(real_img, seg)
        fake_score = model.module.det_img(fake_img, seg)

        # Loss
        vgg_loss = vgg_losser(real_img, fake_img)
        gan_floss = gan_flosser(real_score, fake_score)
        gan_loss = gan_losser(fake_score[-1], True, True)

        if model.module.is_encode:
            kld_losser = KLD_Loss()
            kld_loss = kld_losser(mu, log_s)
            gen_loss = gan_loss + 0.05*kld_loss + 10*vgg_loss + 10*gan_floss
        else:
            gen_loss = gan_loss + 10*vgg_loss + 10*gan_floss

        optimizer_g.zero_grad()
        gen_loss.backward()
        optimizer_g.step()

        real_score = model.module.det_img(real_img, seg)
        fake_score = model.module.det_img(fake_img.detach(), seg)
        real_loss = gan_losser(real_score[-1], True)
        fake_loss = gan_losser(fake_score[-1], False)
        det_loss = (real_loss + fake_loss) * 0.5
        optimizer_d.zero_grad()
        det_loss.backward()
        optimizer_d.step()

        if rank == 0:
            if model.module.is_encode:
                print(f"\rcur_batch: {batch_idx+1}/{len(train_loader)}; gen_loss:[kld:{0.05*kld_loss.item():.2f}; gan:{gan_loss.item():.2f}; gan_f:{10*gan_floss.item():.2f}; vgg:{10*vgg_loss.item():.2f}; all:{gen_loss.item():.2f}]; det_loss: {det_loss.item():.2f}", end='')
            else:
                print(f"\rcur_batch: {batch_idx+1}/{len(train_loader)}; gen_loss:[gan:{gan_loss.item():.2f}; gan_f:{10*gan_floss.item():.2f}; vgg:{10*vgg_loss.item():.2f}; all:{gen_loss.item():.2f}]; det: {det_loss.item():.2f}", end='')
            

def main(rank, world_size, args):
    setup(rank, world_size)

    # model
    model = Model()
    start_epoch = 1
    # model.load("./trained_models/15")
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    parameter_g = []
    parameter_d = []
    for parameter in model.named_parameters():
        if parameter[0].split('.')[1]=="net_d":
            parameter_d.append(parameter[1])
        else:
            parameter_g.append(parameter[1])

    # data
    supercategory = "sports"
    # supercategory = ""
    dataset_path = "/home/ubuntu/spade/dataset"
    train_config = "/home/ubuntu/spade/dataset/annotations/instances_train2017.json"
    
    img_list, label_list, inst_list = CocoHandler(dataset_path, train_config, supercategory).get_all_list()
    if model.module.is_clip:
        feature_list = np.load("./train_feature.npy")
        k_means = KMeans(128)
        k_means.fit(feature_list)
        train_dataset = CocoDataset(img_list, label_list, inst_list, feature_list=feature_list)
    else:
        train_dataset = CocoDataset(img_list, label_list, inst_list)
    
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    train_kwargs = {'batch_size': args.batch_size, 'sampler': train_sampler}
    cuda_kwargs = {'num_workers': 4,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)


    vgg_losser = VGG_Loss().to(rank)
    
    # lr & optim
    optimizer_g = optim.Adam(parameter_g, lr=1e-4, betas=(0, 0.999))
    optimizer_d = optim.Adam(parameter_d, lr=4e-4, betas=(0, 0.999))
    # scheduler_g = LinearLR(optimizer_g,start_factor=1,end_factor=0.1,total_iters=25)
    # scheduler_d = LinearLR(optimizer_d,start_factor=1,end_factor=0.1,total_iters=25)
    # scheduler_g = StepLR(optimizer_g, step_size=30, gamma=args.gamma)
    # scheduler_d = StepLR(optimizer_d, step_size=30, gamma=args.gamma)
    # scheduler_g = MultiStepLR(optimizer_g, [10, 30, 80], gamma=0.5)
    # scheduler_d = MultiStepLR(optimizer_d, [60], gamma=0.5)

    for epoch in range(start_epoch, args.epochs+1):
        train(rank, model, epoch, train_loader, train_sampler, vgg_losser, optimizer_g, optimizer_d)
        # scheduler_g.step()
        # scheduler_d.step()
        if rank == 0:
            print(f"\nEpoch {epoch} finished!\n")
            eval(rank, model, model.module.is_encode, True, k_means)

        if args.is_save and epoch % args.save_frequence == 0 and rank == 0:
            save_path = f"./trained_models/{epoch}"
            if rank==0:
                model.module.save(save_path)

    dist.barrier()
    cleanup()


os.environ['CUDA_VISIBLE_DEVICES']="4"
if __name__ == "__main__":
    args = get_base_args()
    torch.manual_seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
