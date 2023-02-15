import os
import torch
from mvssnetdataset import MVSSNetDataset
from torch.utils.data import DataLoader
from mvssnet import get_mvss
from tool.averagemeter import AverageMeter
from tool.auc import cal_auc
from tool.iou import cal_iou
from tool.f1 import cal_f1
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from tool.Dice_loss import DiceLoss
from torch.utils.tensorboard import SummaryWriter

def train(args):
    # mvssnet_path = "/data/liaogy/mvssnet_casia.pt"
    model = get_mvss(backbone='resnet50', pretrained_base=True, nclass=1, sobel=True, constrain=True, n_input=3,is_plus=True)

    # 模型结构可视化
    # writer = SummaryWriter(comment="grap")
    # writer.add_graph(
    #     model = model,
    #     input_to_model=torch.rand(1, 1, 512, 512)
    # )
    # writer.close()

    # checkpoint = torch.load(mvssnet_path, map_location='cpu')
    # model.load_state_dict(checkpoint, strict=True)
    if len(args.multi_gpu) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()

    # ./././ continue to train, if load./././ #
    load_epoch = 0
    if args.load:
        save_file = torch.load(os.path.join(os.path.join(args.weight_save_dir, args.model + args.suffix),
                                        'experiment_{}_'.format(args.flag) + 'epoch_70.pth'), map_location='cpu')

        # save_file = torch.load(os.path.join(os.path.join(args.weight_save_dir, args.model+args.suffix),
        #                                     'last_epoch.pth'), map_location='cpu')
        load_epoch = save_file['epoch']
        model.load_state_dict(save_file['state_dict'])

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.load:
        optimizer.load_state_dict(save_file['optimizer'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=10, verbose=True)
    criterion = nn.BCELoss()
    criterion_dice = DiceLoss()


    train_dataset = MVSSNetDataset(root=args.root, txt=args.txt, val_txt=args.val_txt, mode='train', is_used_edge=True, size=args.size,
                                    seed=args.seed)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bz, num_workers=args.num_workers, shuffle=True)

    val_dataset = MVSSNetDataset(root=args.root, txt=args.txt, val_txt=args.val_txt, mode='val', is_used_edge=False, size=args.size,
                                  seed=args.seed)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    # ./././ train ./././ #
    losses = AverageMeter()
    max_val_auc = 0.
    if args.flag == 1:
        print("Now start experiment1 without three loss!")
    elif args.flag == 2:
        print("Now start experiment1 with three loss!")
    elif args.flag == 3:
        print("Now start experiment2 with three loss and ConvGem!")

    for epoch in range(load_epoch, args.max_epoches):
        model.train()
        losses.reset()
        t = tqdm(train_loader)
        if args.flag == 1:
            for idx, (forgery, mask) in enumerate(t):
                mask = np.expand_dims(mask, axis=1)
                mask = torch.from_numpy(mask)
                forgery, mask = forgery.cuda(), mask.cuda()
                optimizer.zero_grad()
                edge, outputs = model(forgery, epoch+1)
                print("out.shape, mask.shape", outputs.shape, mask.shape)
                loss = criterion(outputs, mask)
                loss.backward()
                optimizer.step()
                losses.update(loss.detach().cpu().numpy())
                t.set_description('epoch:{} | losses:{}'.format(epoch + 1, losses.avg))
            scheduler.step(losses.avg)

        if args.flag == 2:
            for idx, (image, mask, edge_mask, label) in enumerate(t):
                image, mask, edge_mask, label = image.cuda(), mask.cuda(), edge_mask.cuda(), label.cuda()
                label = label.view(len(label), -1)
                optimizer.zero_grad()
                edge, outputs = model(image, epoch+1)
                loss_seg = criterion_dice(outputs, mask)  # pixel loss
                loss_edge = criterion_dice(edge, edge_mask)  # edge loss
                Gx = []
                batchsize = 0
                for out in outputs:
                    Gx.append(torch.max(out).to(torch.float64))
                    batchsize += 1
                Gx = torch.tensor(Gx).cuda()
                Gx = Gx.view(batchsize, -1)   # Gx.shape=[4,1]
                loss_clf = criterion(Gx, label)
                loss = 0.16 * loss_seg + 0.04 * loss_clf + 0.8 * loss_edge
                loss.backward()
                optimizer.step()
                losses.update(loss.detach().cpu().numpy())
                t.set_description('epoch:{} | losses:{}'.format(epoch + 1, losses.avg))
            scheduler.step(losses.avg)

        if args.flag == 3:
            for idx, (image, mask, edge_mask, label) in enumerate(t):
                image, mask, edge_mask, label = image.cuda(), mask.cuda(), edge_mask.cuda(), label.cuda()
                # label = label.view(len(label), -1)
                optimizer.zero_grad()
                edge, outputs, clf = model(image, epoch+1)
                loss_seg = criterion_dice(outputs, mask)  # pixel loss
                loss_edge = criterion_dice(edge, edge_mask)  # edge loss
                clf = clf.reshape((len(clf), 1))
                loss_clf = criterion(clf.float(), label.float())
                loss = 0.16 * loss_seg + 0.04 * loss_clf + 0.8 * loss_edge
                loss.backward()
                optimizer.step()
                losses.update(loss.detach().cpu().numpy())
                t.set_description('epoch:{} | losses:{}'.format(epoch + 1, losses.avg))
            scheduler.step(losses.avg)
        # ./././ val ./././ #
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_iou = AverageMeter()
            val_f1 = AverageMeter()
            val_auc = AverageMeter()
            t = tqdm(val_loader)
            with torch.no_grad():
                for idx, (forgery, mask) in enumerate(t):
                    forgery, mask = forgery.cuda(), mask.cuda()
                    if args.flag == 3:
                        edge, outputs, _ = model(forgery, epoch+1)
                    else:
                        edge, outputs = model(forgery, epoch + 1)
                    trans_masks = mask.detach().cpu().numpy()
                    trans_outputs = outputs.detach().cpu().numpy()
                    val_auc.update(cal_auc(trans_outputs, trans_masks))
                    # trans_outputs_image = np.where(trans_outputs > 0.5, 1, 0)
                    val_iou.update(cal_iou(trans_outputs, trans_masks))
                    val_f1.update(cal_f1(trans_outputs, trans_masks))
                    t.set_description('epoch:{} | iou_pixel:{} | f1_pixel:{} | auc:{}'.format(epoch + 1, val_iou.avg, val_f1.avg, val_auc.avg))

        if val_auc.avg > max_val_auc:
            max_val_auc = val_auc.avg
            save_states_path = os.path.join(os.path.join(args.weight_save_dir, args.model+args.suffix),
                   'experiment_{}_'.format(args.flag)+'epoch_{}.pth'.format(epoch + 1))
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'F1': val_f1.avg,
                'AUC': val_auc.avg,
                'IoU': val_iou.avg,
                'args': args,
            }
            torch.save(states, save_states_path)
            print('Current auc:', val_auc.avg, 'Best acc:', max_val_auc)







