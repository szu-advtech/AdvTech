import torch
import argparse
import torchvision.transforms as transforms
import os
from data_LDL import Emotion_LDL
from models import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from adamw import AdamW
# from torch.autograd import Variable
import utils
import numpy as np
from torchvision import models
from tensorboardX import SummaryWriter
# from models.TL import Triplet
from models.MSE_loss_theta import MSE_Loss_theta
from models.Polarloss import PolarLoss
# from models.CE_loss_weighed import CELoss_weighed
from models.polar_coordinates import Polar_coordinates
from evaluation_metric import Evaluation_metrics
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# random seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True            # cudnn

def main():
    # Parameters
    parser = argparse.ArgumentParser(description='PyTorch Emotion_LDL CNN Training')
    # parser.add_argument('--img_path', type=str, default='/home/tingting/code/twitter_ldl/images/')
    # parser.add_argument('--train_csv_file', type=str,
    #                     default='/home/tingting/code/twitter_ldl/twitter_truth_train.csv')
    # parser.add_argument('--test_csv_file', type=str,
    #                     default='/home/tingting/code/twitter_ldl/twitter_truth_test.csv')
    parser.add_argument('--img_path', type=str, default='/home/tingting/code/FI_dataset/image/')
    parser.add_argument('--train_csv_file', type=str,
                        default='/home/tingting/code/FI_dataset/fi_csv_train.csv')
    parser.add_argument('--test_csv_file', type=str,
                        default='/home/tingting/code/FI_dataset/fi_csv_test.csv')
    parser.add_argument('--ckpt_path', type=str, default='/home/tingting/code/cc_4/ckpts_twi_kl_p')
    parser.add_argument('--model', type=str, default='ResNet_50', help='CNN architecture')
    parser.add_argument('--dataset', type=str, default='Emotion_LDL', help='Dataset')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkposint')

    parser.add_argument('--if_decay', default=1, type=int, help='decay lr every 5 epochs')
    parser.add_argument('--decay', default=0.1, type=float, help='decay value every 5 epochs')
    parser.add_argument('--start', default=10, type=float, help='decay value every 5 epochs')
    parser.add_argument('--every', default=10, type=float, help='decay value every 5 epochs')
    parser.add_argument('--lr_adam', default=1e-5, type=float, help='learning rate for adam|5e-4|1e-5|smaller')
    parser.add_argument('--lr_sgd', default=5e-4, type=float,  help='learning rate for sgd|1e-3|5e-4')
    parser.add_argument('--wd', default=5e-5, type=float, help='weight decay for adam|1e-4|5e-5')
    parser.add_argument('--optimizer', default='adamw', type=str, help='sgd|adam|adamw')
    parser.add_argument('--gpu', default=5, type=int, help='0|1|2|3')

    parser.add_argument('--seed', default=2088, type=int, help='66just a random seed')
    opt = parser.parse_args()

# set gpu device
    torch.cuda.set_device(opt.gpu)

    set_seed(seed=opt.seed)

    writer = SummaryWriter()  

    best_test_acc = 0
    best_test_acc_epoch = 0

    start_epoch = 0

    total_epoch = 55

    path = os.path.join(opt.dataset + '_' + opt.model)

    # Data
    print('==> Preparing data..')

    transform_train = transforms.Compose([
        transforms.Resize(480),  
        transforms.RandomCrop(448), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
            transforms.Resize(480),
            transforms.RandomCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    trainset = Emotion_LDL(csv_file=opt.train_csv_file, root_dir=opt.img_path, transform=transform_train)
    testset= Emotion_LDL(csv_file=opt.test_csv_file, root_dir=opt.img_path, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=8)

    # Model
    if opt.model == 'ResNet_50':
        base_model = models.resnet50(pretrained=True) ###
        net = model_baseline(base_model)
    elif opt.model == 'ResNet_101':
        base_model = models.resnet101(pretrained=True) ###
        net = model_baseline(base_model)
    elif opt.model == 'VGG_19':
        base_model = models.vgg19(pretrained=True)
        net = model_baseline(base_model)

    param_num = 0
    for param in net.parameters():
        param_num = param_num + int(np.prod(param.shape))

    print('==> Trainable params: %.2f million' % (param_num / 1e6))

    if opt.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('/home/tingting/code/cc_4/ckpts_twi_kl_p/epoch-19.pkl', map_location="cuda:0")
        net.load_state_dict(checkpoint)
    else:
        print('==> Building model..')
    if torch.cuda.is_available():
        net.cuda()

    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()
    KLloss = nn.KLDivLoss(size_average=False, reduce=True)
    MSELoss_theta = MSE_Loss_theta()
    Polarloss = PolarLoss()

    if torch.cuda.is_available():
        CEloss = CEloss.cuda()
        MSEloss = MSEloss.cuda()
        KLloss = KLloss.cuda()

    if opt.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr_adam, weight_decay=opt.wd)
    elif opt.optimizer == 'adamw':
        optimizer = AdamW(net.parameters(), lr=opt.lr_adam, weight_decay=opt.wd, amsgrad=False)
    elif opt.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=opt.lr_sgd, momentum=0.9, weight_decay=5e-4)

        # Data
    print('==> Preparing data..')

    for epoch in range(start_epoch, total_epoch):
        print(epoch)
        train(epoch, opt, net, writer, trainloader, optimizer, KLloss, MSEloss, Polar_coordinates, MSELoss_theta, Polarloss)
        best_test_acc, best_test_acc_epoch = test(epoch, net, writer, testloader, KLloss, best_test_acc,
                                                best_test_acc_epoch, path, MSEloss, Polar_coordinates, MSELoss_theta, Polarloss)

        print("best_test_acc: %0.3f" % best_test_acc)
        print("best_test_acc_epoch: %d" % best_test_acc_epoch)



# Training
def train(epoch, opt, net, writer, trainloader, optimizer, KLloss, MSEloss, Polar_coordinates, MSELoss_theta, Polarloss):
    print('\nEpoch: %d' % epoch)
    global train_acc
    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    train_Dist_1 = 0
    train_Dist_2 = 0
    train_Dist_3 = 0
    train_Dist_4 = 0
    train_Sim_1 = 0
    train_Sim_2 = 0
    correct = 0
    total = 0

    if opt.if_decay == 1:
        if epoch >= opt.start:
            frac = (epoch - opt.start) // opt.every + 1  # round
            decay_factor = opt.decay ** frac  # how many times we have this decay

            if opt.optimizer == 'adam':
                current_lr = opt.lr_adam
            elif opt.optimizer == 'adamw':
                current_lr = opt.lr_adam
            elif opt.optimizer == 'sgd':
                current_lr = opt.lr_sgd
            current_lr = current_lr * decay_factor # new learning rate
            for rr in range(len(optimizer.param_groups)):
                utils.set_lr(optimizer, current_lr, rr)  # set the decayed learning rate
        else:
            if opt.optimizer == 'adam':
                current_lr = opt.lr_adam
            elif opt.optimizer == 'adamw':
                current_lr = opt.lr_adam
            elif opt.optimizer == 'sgd':
                current_lr = opt.lr_sgd
        print('learning_rate: %s' % str(current_lr))

    for batch_idx, data in enumerate(trainloader):
        images = data['image']
        dist_emo = data['dist_emo']

        if torch.cuda.is_available():
            images = images.cuda()
            dist_emo = dist_emo.cuda()

        optimizer.zero_grad()
        net.train()
        emo = net(images)
   
        theta_emo, _= Polar_coordinates(emo)
        theta_dist_emo, r_dist_emo = Polar_coordinates(dist_emo)

        loss1 = KLloss(emo.log(), dist_emo)
        loss2 = MSELoss_theta(theta_emo, theta_dist_emo, r_dist_emo)
        loss3 = Polarloss(theta_emo, theta_dist_emo, r_dist_emo)
        loss = loss1 

        loss.backward()
        optimizer.step()

        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        train_loss3 += loss3.item()
        train_loss += loss.item()

        Dis_1, Dis_2, Dis_3, Dis_4, Sim_1, Sim_2 = Evaluation_metrics(emo, dist_emo)
        train_Dist_1 += Dis_1
        train_Dist_2 += Dis_2
        train_Dist_3 += Dis_3
        train_Dist_4 += Dis_4
        train_Sim_1 += Sim_1
        train_Sim_2 += Sim_2

        _, predicted = torch.max(emo.data, 1)
        _, labeled = torch.max(dist_emo.data, 1)
        total += dist_emo.size(0)
        correct += predicted.eq(labeled.data).cpu().sum().numpy()
        train_acc = 100. * correct / total

        utils.progress_bar(batch_idx, len(trainloader),
                        'Loss1: %.3f Loss2: %.3f Loss3: %.3f Loss: %.3f '
                        '| Chebyshev: %.3f Clark: %.3f Canberra: %.3f KL: %.3f Cosine: %.3f Inter: %.3f Acc: %.3f%%'
                        % (train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                            train_loss3 / (batch_idx + 1), train_loss / (batch_idx + 1),
                            train_Dist_1 / (batch_idx + 1), train_Dist_2 / (batch_idx + 1),
                            train_Dist_3 / (batch_idx + 1), train_Dist_4 / (batch_idx + 1),
                            train_Sim_1 / (batch_idx + 1), train_Sim_2 / (batch_idx + 1), train_acc))

    writer.add_scalar('data/Train_Loss', train_loss, epoch)
    writer.add_scalar('data/Train_Loss1', train_loss1 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Train_Loss2', train_loss2 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Train_Loss3', train_loss3 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Train_Chebyshev', train_Dist_1 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Train_Clark', train_Dist_2 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Train_Canberra', train_Dist_3 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Train_KL', train_Dist_4 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Train_Cosine', train_Sim_1 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Train_Inter', train_Sim_2 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Train_Acc', train_acc, epoch)
    print('==> Saving model...')
    torch.save(net.state_dict(), os.path.join(opt.ckpt_path, 'epoch-%d.pkl' % epoch))

# Test
def test(epoch, net, writer, testloader, KLloss, best_test_acc, best_test_acc_epoch, path, MSEloss, Polar_coordinates, MSELoss_theta, Polarloss):
    global test_acc

    test_loss1 = 0
    test_loss2 = 0
    test_loss3 = 0
    test_Dist_1 = 0
    test_Dist_2 = 0
    test_Dist_3 = 0
    test_Dist_4 = 0
    test_Sim_1 = 0
    test_Sim_2 = 0
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(testloader):
        images = data['image']
        dist_emo = data['dist_emo']

        if torch.cuda.is_available():
            images = images.cuda()
            dist_emo = dist_emo.cuda()

        with torch.no_grad():
            net.eval()
            emo = net(images)
            theta_emo, r_emo = Polar_coordinates(emo)
            theta_dist_emo, r_dist_emo = Polar_coordinates(dist_emo)

            loss1 = KLloss(emo.log(), dist_emo)
            loss2 = MSELoss_theta(theta_emo, theta_dist_emo, r_dist_emo)
            loss3 = Polarloss(theta_emo, theta_dist_emo, r_dist_emo)
     
            loss = loss1 

            test_loss1 += loss1.item()
            test_loss2 += loss2.item()
            test_loss3 += loss3.item()
            test_loss += loss.item()

            Dis_1, Dis_2, Dis_3, Dis_4, Sim_1, Sim_2 = Evaluation_metrics(emo, dist_emo)
            test_Dist_1 += Dis_1
            test_Dist_2 += Dis_2
            test_Dist_3 += Dis_3
            test_Dist_4 += Dis_4
            test_Sim_1 += Sim_1
            test_Sim_2 += Sim_2

            _, predicted = torch.max(emo.data, 1)
            _, labeled = torch.max(dist_emo.data, 1)
            total += dist_emo.size(0)
            correct += predicted.eq(labeled.data).cpu().sum().numpy()
            test_acc = 100. * correct / total

            utils.progress_bar(batch_idx, len(testloader),
                            'Loss1: %.3f Loss2: %.3f Loss3: %.3f Loss: %.3f '
                            '| Chebyshev: %.3f Clark: %.3f Canberra: %.3f KL: %.3f Cosine: %.3f Inter: %.3f Acc: %.3f%%'
                            % (test_loss1 / (batch_idx + 1), test_loss2 / (batch_idx + 1),
                                test_loss3 / (batch_idx + 1), test_loss / (batch_idx + 1),
                                test_Dist_1 / (batch_idx + 1), test_Dist_2 / (batch_idx + 1),
                                test_Dist_3 / (batch_idx + 1), test_Dist_4 / (batch_idx + 1),
                                test_Sim_1 / (batch_idx + 1), test_Sim_2 / (batch_idx + 1), test_acc))

    writer.add_scalar('data/Test_Loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_Loss1', test_loss1 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_Loss2', test_loss2 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_Loss3', test_loss3 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_Chebyshev', test_Dist_1 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_Clark', test_Dist_2 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_Canberra', test_Dist_3 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_KL', test_Dist_4 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_Cosine', test_Sim_1 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_Inter', test_Sim_2 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_Acc', test_acc, epoch)

    # Save checkpoint.
    if test_acc > best_test_acc:
        print('==> Finding best acc..')
        state = {
            'net': net.state_dict() if torch.cuda.is_available() else net,
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'test_model.t7'))
        best_test_acc = test_acc
        best_test_acc_epoch = epoch
    return best_test_acc, best_test_acc_epoch

if __name__ == '__main__':
    main()
    print('Finish training')