import argparse
import numpy as np
from model import *
import json
from dataset import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import lr_scheduler

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='Pytorch Training')
parser.add_argument('-t', '--train', default=True, type=bool,
                    help='need train or not')
parser.add_argument('-w', '--weight', default=None, type=str,
                    help='Path to the .pth file to load')
parser.add_argument('-i', '--image', default='data/img.mat', type=str,
                    help='the image use to train model')
parser.add_argument('-c', '--config', default='config.json', type=str,
                    help='Path to the config file')
parser.add_argument('-e', '--epoch', default=3000, type=str,
                    help='Train epoch')
parser.add_argument('-d', '--device', default='cuda:3', type=str,
                    help='training device')
args = parser.parse_args()

config = json.load(open(args.config))

gpus = [0, 1, 2, 3]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

transform = t.Compose([
    t.ToTensor(),
    t.RandomRotation(0.5),
    t.RandomHorizontalFlip(0.5)
])

# device = args.device if 'cuda' in args.device else 'cpu'
# print(device)

unmixing_train_dataset = dataset(is_downsample=False, is_train=True)
unmixing_train_dataloader = DataLoader(dataset=unmixing_train_dataset, batch_size=config['chikusei']['batch_size'],
                                       shuffle=True, drop_last=True)

unmixing_validation_dataset = dataset(is_downsample=False, is_train=False)
unmixing_validation_dataloader = DataLoader(dataset=unmixing_validation_dataset, batch_size=len(unmixing_validation_dataset),
                                            shuffle=True)

dpcn_train_dataset = dataset(is_downsample=True, is_train=True)
dpcn_train_dataloader = DataLoader(dataset=dpcn_train_dataset, batch_size=config['chikusei']['batch_size'],
                                   shuffle=True, drop_last=True)

dpcn_validation_dataset = dataset(is_downsample=False, is_train=False)
dpcn_validation_dataloader = DataLoader(dataset=dpcn_validation_dataset, batch_size=len(dpcn_validation_dataset),
                                        shuffle=True)

linear_unmixing_encoder = Linear_unmixing_encoder(config)
linear_unmixing_encoder = nn.DataParallel(linear_unmixing_encoder.cuda(), device_ids=gpus, output_device=gpus[0])
# linear_unmixing_encoder.load_state_dict(torch.load('./model_weight/intermediate_weight/fusion_rgb_resnet/linear_unmixing_encoder_first.pth'))

# dpcn = DPCN(config)
# dpcn.to(device)
resnet = ResNet(config, ResidualBlock, [2, 2, 2, 2])
resnet = nn.DataParallel(resnet.cuda(), device_ids=gpus, output_device=gpus[0])
# resnet.load_state_dict(torch.load('./model_weight/intermediate_weight/fusion_rgb_resnet/resnet_60epoch_c=40.pth'))

linear_unmixing_decoder = Linear_unmixing_decoder(config)
linear_unmixing_decoder = nn.DataParallel(linear_unmixing_decoder.cuda(), device_ids=gpus, output_device=gpus[0])
# linear_unmixing_decoder.load_state_dict(torch.load('./model_weight/intermediate_weight/fusion_rgb_resnet/linear_unmixing_decoder_first.pth'))

optimizer_linear_unmixing = Adam([{"params": linear_unmixing_encoder.parameters(), },
                                  {"params": linear_unmixing_decoder.parameters()}], lr=0.0001, weight_decay=1e-4)
optimizer_dpcn = Adam(resnet.parameters(), lr=0.0001, weight_decay=1e-4)
optimizer_all = Adam([{"params": linear_unmixing_encoder.parameters()},
                      {"params": linear_unmixing_decoder.parameters()},
                      {"params": resnet.parameters()}], lr=0.0001, weight_decay=1e-4)

scheduler_linear_unmixing = lr_scheduler.StepLR(optimizer_linear_unmixing, step_size=2000)
scheduler_dpcn = lr_scheduler.StepLR(optimizer_dpcn, step_size=2000)

"------------------------------------------------train and validation--------------------------------------------------"
if not args.train:
    pass
else:
    # # 训练unmixing
    print('-------------------------------train start------------------------------------')
    print('train unmixing first')
    last_sam_loss = 0
    for i in range(30):
        sam_loss_list = []
        sam_rgb_list = []
        print('training {}/30 epoch to unmixing'.format(i + 1))
        for item_num, ((img_hsi_x, img_rgb_x), (img_hsi_y, img_rgb_y)) in enumerate(unmixing_train_dataloader):
            optimizer_linear_unmixing.zero_grad()
            img_hsi_x, img_rgb_x = img_hsi_x.cuda(non_blocking=True).float(), img_rgb_x.cuda(non_blocking=True).float()
            img_hsi_y, img_rgb_y = img_hsi_y.cuda(non_blocking=True).float(), img_rgb_y.cuda(non_blocking=True).float()

            img_hsi_x = linear_unmixing_decoder(linear_unmixing_encoder(img_hsi_x))
            img_rgb_x = linear_unmixing_decoder(linear_unmixing_encoder(img_rgb_x))

            l1_loss = loss_function(predict=img_hsi_x, ground_truth=img_hsi_y, mode='l1_loss')

            sam_loss_rgb = loss_function(img_rgb_x, img_rgb_y, mode='sam_loss')
            sam_loss = loss_function(img_hsi_x, img_hsi_y, mode='sam_loss')
            sam_loss_list.append(sam_loss.item())
            sam_rgb_list.append(sam_loss_rgb.item())

            loss_unmixing = l1_loss + 0.1 * sam_loss

            loss_unmixing.backward()
            optimizer_linear_unmixing.step()
            if item_num % 4 == 0:
                print("sam: {}".format(sam_loss), 'sam_rgb: {}'.format(sam_loss_rgb))
        sam_loss_avg = np.array(sam_loss_list).mean()
        sam_rgb_avg = np.array(sam_rgb_list).mean()
        print('epoch: {}, sam_loss_avg: {}, sam_rgb_loss: {}'.format(i + 1, sam_loss_avg, sam_rgb_avg))
        print()
        if abs(sam_loss_avg - last_sam_loss) < 0.1:
            print('end before because of sam_loss')
            break
        else:
            last_sam_loss = sam_loss_avg

    print('Training unmixing end')

    # 冻结线性解混模型中的参数
    for name, values in linear_unmixing_encoder.named_parameters():
        values.requires_grad = False

    for name, values in linear_unmixing_decoder.named_parameters():
        values.requires_grad = False

    torch.save(linear_unmixing_encoder.state_dict(),
               './model_weight/intermediate_weight/fusion_rgb_resnet/linear_unmixing_encoder_first.pth')
    torch.save(linear_unmixing_decoder.state_dict(),
               './model_weight/intermediate_weight/fusion_rgb_resnet/linear_unmixing_decoder_first.pth')
    # #将unmixing中的decoder参数共享给DPCN后的decoder
    # dict_decoder = {} #用于load_state
    # decoder_name_list = ['decoder.weight', 'decoder.bias']  #层的名字，用于取values
    #
    # for name,values in linear_unmixing.named_parameters():
    #     if name in decoder_name_list:
    #         dict_decoder[name] = values
    #
    # linear_unmixing_decoder.load_state_dict(dict_decoder)
    # for name,values in linear_unmixing_decoder.named_parameters():
    #     values.requires_grad = False

    # 训练主体网络DPCN
    print()
    print('train resnet')
    last_sam_loss = 0
    count = 0
    for epoch in range(args.epoch):
        sam_loss_list = []
        pnsr_list = []
        egras_list = []
        rmse_list = []
        for step, (img_x, img_y) in enumerate(dpcn_train_dataloader):

            optimizer_dpcn.zero_grad()
            img_x, img_y = img_x.cuda(non_blocking=True).float(), img_y.cuda(non_blocking=True).float()

            img_x = linear_unmixing_encoder(img_x)
            img_x = resnet(img_x)
            img_x = linear_unmixing_decoder(img_x)

            l1_loss = loss_function(predict=img_x, ground_truth=img_y, mode='l1_loss')
            sam_loss = loss_function(img_x, img_y, mode='sam_loss')
            sam_loss_list.append(sam_loss.item())
            loss_dpcn = l1_loss + 0.1 * sam_loss

            pnsr_ = pnsr(img_x, img_y)
            pnsr_list.append(pnsr_.item())
            egras_ = egras(img_x, img_y, scale_factor=4)
            egras_list.append(egras_.item())
            rmse_ = rmse(img_x, img_y)
            rmse_list.append(rmse_.item())

            loss_dpcn.backward()
            optimizer_dpcn.step()
            if step % 3 == 0:
                print('epoch: {}/{}'.format(epoch, args.epoch),
                      'batch: {}/{}'.format((step), len(dpcn_train_dataloader)),
                      'PNSR: {}'.format(pnsr_.item()),
                      'egras: {}'.format(egras_.item()),
                      'RMSE: {}'.format(rmse_.item()),
                      'SAM_loss: {}'.format(sam_loss),
                      'loss_dpcn: {}'.format(loss_dpcn))

        sam_loss_avg = np.array(sam_loss_list).mean()
        print("training DPCN -- ", 'epoch: {}/3000'.format(epoch + 1),
              'PNSR_avg: {}'.format(np.array(pnsr_list).mean()),
              'egras_avg: {}'.format(np.array(egras_list).mean()), 'RMSE_avg: {}'.format(np.array(rmse_list).mean())
              , 'sam_loss_avg: {}'.format(sam_loss_avg))
        print()
        if epoch % 20 == 0 and epoch != 0:
            torch.save(resnet.state_dict(), './model_weight/intermediate_weight/resnet_{}epoch_c=40.pth'.format(epoch))

        if abs(sam_loss_avg - last_sam_loss) < 0.1:
            count += 1
            if count == 4:
                break
            else:
                last_sam_loss = sam_loss_avg
                pass
        else:
            last_sam_loss = sam_loss_avg
            count = 0
    # 整个网络
    for name, values in linear_unmixing_encoder.named_parameters():
        values.requires_grad = True

    for name, values in linear_unmixing_decoder.named_parameters():
        values.requires_grad = True

    print()
    print('training all')
    last_sam_loss = 0
    for epoch in range(args.epoch):
        pnsr_list = []
        egras_list = []
        rmse_list = []
        sam_loss_list = []
        for step, (img_x, img_y) in enumerate(dpcn_train_dataloader):
            optimizer_all.zero_grad()
            img_x, img_y = img_x.cuda(non_blocking=True).float(), img_y.cuda(non_blocking=True).float()

            img_x = linear_unmixing_encoder(img_x)
            img_x = resnet(img_x)
            img_x = linear_unmixing_decoder(img_x)

            pnsr_ = pnsr(img_x, img_y)
            pnsr_list.append(pnsr_.item())
            egras_ = egras(img_x, img_y, scale_factor=4)
            egras_list.append(egras_.item())
            rmse_ = rmse(img_x, img_y)
            rmse_list.append(rmse_.item())

            l1_loss = loss_function(predict=img_x, ground_truth=img_y, mode='l1_loss')
            sam_loss = loss_function(img_x, img_y, mode='sam_loss')
            sam_loss_list.append(sam_loss.item())
            loss_all = l1_loss + 0.2 * sam_loss

            print('batch: {}/{}'.format((step + 1), len(dpcn_train_dataloader)),
                  'PNSR: {}'.format(pnsr_),
                  'egras: {}'.format(egras_),
                  'RMSE: {}'.format(rmse_),
                  'SAM_loss: {}'.format(sam_loss),
                  'loss_dpcn: {}'.format(loss_all))

            loss_all.backward()
            optimizer_all.step()

        sam_loss_avg = np.array(sam_loss_list).mean()

        print('epoch: {}/3000'.format(epoch + 1), 'PNSR_avg: {}'.format(np.array(pnsr_list).mean()),
              'egras_avg: {}'.format(np.array(egras_list).mean()), 'RMSE_avg: {}'.format(np.array(rmse_list).mean()),
              'sam_loss_avg: {}'.format(sam_loss_avg))
        print()
        if (epoch + 1) % 20 == 0 and epoch != 0:
            torch.save(resnet.state_dict(),
                       './model_weight/intermediate_weight/resnet_all_{}epoch_c=40.pth'.format(epoch))
            torch.save(linear_unmixing_encoder.state_dict(),
                       './model_weight/intermediate_weight/linear_unmixing_encoder_all_{}epoch_c=40.pth'.format(epoch))
            torch.save(linear_unmixing_decoder.state_dict(),
                       './model_weight/intermediate_weight/linear_unmixing_decoder_all_{}epoch_c=40.pth'.format(epoch))
        if abs(sam_loss_avg - last_sam_loss) < 0.1:
            count += 1
            if count == 4:
                break
            else:
                last_sam_loss = sam_loss_avg
                pass
        else:
            last_sam_loss = sam_loss_avg
            count = 0
    torch.save(linear_unmixing_encoder.state_dict(), './model_weight/linear_unmixing_encoder_c=40.pth')
    torch.save(linear_unmixing_decoder.state_dict(), './model_weight/linear_unmixing_decoder_c=40.pth')
    torch.save(resnet.state_dict(), './model_weight/resnet_c=40.pth')
