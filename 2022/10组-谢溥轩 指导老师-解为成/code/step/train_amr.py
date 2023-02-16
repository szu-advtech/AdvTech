import logging
import os
import time
import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from misc import torchutils
from misc.pyutils import AverageMeter
from config import cfg
from amr_utils.logger import setup_logger
from amr_utils import dist_util
from amr_utils.build import make_optimizer, make_lr_scheduler
from amr_utils.checkpoint import CheckPointer
from amr_utils.metric_logger import MetricLogger
from net import resnet50_amr
import voc12.dataloader


def validate(model, data_loader):
    logger = logging.getLogger('AMR.trainer')
    logger.info("Start validating...")
    val_loss_meter = AverageMeter('cls_loss')
    model.eval()
    label_loss = nn.MultiLabelSoftMarginLoss()
    with torch.no_grad():
        for pack in data_loader:
            img = pack['img'].cuda(non_blocking=True)
            label = pack['label'].cuda(non_blocking=True)

            spotlight_logits, spotlight_cam, compensation_logits, compensation_cam = model(img)

            L_spotlight_label = label_loss(spotlight_logits, label)
            L_compensation_label = label_loss(compensation_logits, label)
            L_cps = torch.mean(torch.abs(spotlight_cam - compensation_cam))

            cls_loss = 0.5 * L_spotlight_label + 0.5 * L_compensation_label
            total_loss = cls_loss + 0.05 * L_cps

            val_loss_meter.add({'total_loss': total_loss, 'cls_loss': cls_loss.item()})

    logger.info("total_loss:{},cls_loss:{}".format(
        val_loss_meter.pop('total_loss'),
        val_loss_meter.pop('cls_loss')
    ))
    model.train()
    return


def run(cfg, args):
    # set logger
    logger = setup_logger("AMR.trainer", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info(args)
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    logger = logging.getLogger('AMR.trainer')

    model = resnet50_amr.Net()
    pretrain_dict = torch.load("/data2/xiepuxuan/code/AMR/res50_cam_new.pth")
    new_dict = {}
    for k, v in pretrain_dict.items():
        if "resnet50" in k:
            new_dict[k.replace("resnet50", "resnet50_spotlight")] = v
    model_dict = model.state_dict()
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    lr = cfg.SOLVER.LR

    # Train and validate dataset
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              resize_long=(320, 640), hor_flip=True,
                                                              crop_size=512, crop_method="random")
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # set optimizer,scheduler and checkpointer
    optimizer = make_optimizer(cfg, model, lr)
    max_step = (len(train_dataset) // args.batch_size) * args.num_epochs
    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': lr, 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': param_groups[1], 'lr': 10 * lr, 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': param_groups[2], 'lr': lr, 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
    ], lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY, max_step=max_step)
    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0
    checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    # start training
    logger.info("Start training ......")
    model.train()
    meters = MetricLogger()
    max_iter = len(train_dataset)
    start_iter = 0
    start_training_time = time.time()
    end = time.time()

    label_criterion = nn.MultiLabelSoftMarginLoss()
    for epoch in range(args.num_epochs):
        logger.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        for iteration, pack in enumerate(train_data_loader, start_iter):
            iteration += 1
            images = pack['img'].to(device)
            label = pack['label'].to(device)
            spotlight_logits, spotlight_cam, compensation_logits, compensation_cam = model(images)

            # two branch loss function
            L_spotlight_label = label_criterion(spotlight_logits, label)
            L_compensation_label = label_criterion(compensation_logits, label)
            L_cps = torch.mean(torch.abs(spotlight_cam - compensation_cam))

            loss_dict = dict(
                cls_loss=(L_spotlight_label + L_compensation_label) / 2,
                cps_loss=0.05 * L_cps
            )

            # compute loss
            loss = sum(loss for loss in loss_dict.values())
            meters.update(total_loss=loss, **loss_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # compute time of running a batch
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)

            # output log information
            if iteration % args.log_step == 0:
                eta_seconds = meters.time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if device == "cuda":
                    logger.info(
                        meters.delimiter.join([
                            "iter: {iter:06d}",
                            "lr: {lr:.5f}",
                            '{meters}',
                            "eta: {eta}",
                            'mem: {mem}M',
                        ]).format(
                            iter=iteration,
                            lr=optimizer.param_groups[0]['lr'],
                            meters=str(meters),
                            eta=eta_string,
                            mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                        )
                    )
                else:
                    logger.info(
                        meters.delimiter.join([
                            "iter: {iter:06d}",
                            "lr: {lr:.5f}",
                            '{meters}',
                            "eta: {eta}",
                        ]).format(
                            iter=iteration,
                            lr=optimizer.param_groups[0]['lr'],
                            meters=str(meters),
                            eta=eta_string,
                        )
                    )

            # save model when reach save_step
            if iteration % args.save_step == 0:
                checkpointer.save("amr_model_epoch{:05d}_{:06d}".format(epoch, iteration), **arguments)

            # evaluate model when reach eval_step
            if args.eval_step > 0 and iteration % args.eval_step == 0 and iteration != max_iter:
                validate(model, val_data_loader)

    # save final model
    checkpointer.save("final_model", **arguments)
    torch.cuda.empty_cache()
