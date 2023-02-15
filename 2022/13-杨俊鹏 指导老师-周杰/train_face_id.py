import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
#from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import math
import os
from torchvision import transforms
from models import RSCNN_SSN_Cls as RSCNN_SSN
from models import ContrastiveLoss
from data import Bosphorus, BU3DFE, KinectLQ_train
import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml
#import visdom

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 
######################################################################################
"""
vis = visdom.Visdom(port='8098')
def plot_current_errors(plot_data): # plot_data: {'X':list, 'Y':list, 'legend':list}
    vis.line(
        X=np.stack([np.array(plot_data['X'])]*len(plot_data['legend']),1),
        Y=np.array(self.plot_data['Y']),
        opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)
plot_data = {'X':[], 'Y':[], 'legend':['train_loss']}
"""
######################################################################################
parser = argparse.ArgumentParser(description='Relation-Shape CNN Face ID Classification Training')
parser.add_argument('--config', default='cfgs/config_ssn_face_id.yaml', type=str)

def data_augmentation(point_cloud):
    """point cloud data augmentation using function from data_utils.py"""

    PointcloudRandomInputDropout = d_utils.PointcloudRandomInputDropout()
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()
    PointcloudJitter = d_utils.PointcloudJitter(clip=0.02)
    angle = math.pi/2.0
    
    PointcloudRotatebyRandomAngle_y = d_utils.PointcloudRotatebyRandomAngle(rotation_angle=angle, 
        axis=np.array([0.0, 1.0, 0.0]))
    PointcloudRotatebyRandomAngle_x = d_utils.PointcloudRotatebyRandomAngle(rotation_angle=angle, 
        axis=np.array([1.0, 0.0, 0.0]))

    transform_func = {0: lambda x: x,
            1:PointcloudScaleAndTranslate,
            #2:PointcloudJitter,
            3:PointcloudRotatebyRandomAngle_x,
            4:PointcloudRotatebyRandomAngle_y
            }

    method_num = 1

    func_id = np.array(list(transform_func.keys()))
    pro = [0.2] # probability of each transform function
    other_pro = (1-pro[0])/float(len(transform_func)-1)
    pro.extend([other_pro]*(func_id.shape[0]-1))

    func_use_id = np.random.choice(func_id, method_num, replace=False, p=pro)
    
    if 0 in list(func_use_id):
        return PointcloudRandomInputDropout(point_cloud)

    for idx in func_use_id:
        point_cloud = transform_func[idx](point_cloud)
    point_cloud = PointcloudRandomInputDropout(point_cloud)
    return point_cloud



def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")
    
    try:
        os.makedirs(args.save_path)
    except OSError:
        pass
    
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    
    #train_dataset = BU3DFE(num_points=args.num_points, root=args.data_root, transforms=train_transforms, train=True, task='id')
    train_dataset = KinectLQ_train(num_points=args.num_points+2000, root=args.data_root, transforms=train_transforms, train=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers = int(args.workers),
        drop_last=True
    )

    #test_dataset = Bosphorus(num_points=args.num_points, root=args.data_root, transforms=test_transforms, train=True, task='id')
    #test_dataset = KinectLQ_train(num_points=args.num_points, root=args.data_root, transforms=test_transforms, train=False, task='id')
    test_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=int(args.workers),
        drop_last=False
    )
    
    model = RSCNN_SSN(num_classes=args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    #model = RSCNN_MS(num_classes=args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    #model = dgcnn.DGCNN(output_channels = args.num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>=2:
        model = nn.DataParallel(model, device_ids=[0])
        model.to(device)
    elif torch.cuda.device_count()==1:
        model.cuda()
    # optimizer = optim.Adam(
    #     model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, momentum=0.9)

    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    #lr_scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-4)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)
    
    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))

    criterion = nn.CrossEntropyLoss()
    # glob_criterion = ContrastiveLoss.ContrastiveLossWithinBatch(margin=1.0, method="L2") # 0.3
    #criterion_cent = center_loss.CenterLoss(num_classes=args.num_classes, feat_dim=512, use_gpu=True) # center loss
    #optimizer_cent = optim.Adam(criterion_cent.parameters(), lr=0.01)
    
    num_batch = len(train_dataset)/args.batch_size
    
    # training
    train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)
    

def train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    #global g_acc 
    global g_loss
    g_loss = 16   # only save the model whose acc > 0.9
    batch_count = 0
    model.train()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_dataloader, 0):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
               bnm_scheduler.step(epoch-1)
            points, target = data
            points, target = points.cuda(), target.cuda()
            points, target = Variable(points), Variable(target)
            
            # fastest point sampling
            fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            
            # augmentation
            points = data_augmentation(points)
            
            optimizer.zero_grad()
            #optimizer_cent.zero_grad()
            # watch gradient
            """
            def loss_hook(loss, gradin, gradout):
                with open("log/inter_data/AngleLinear_gradient.txt", 'a') as gf:
                    gf.write("Epoch {} ".format(epoch))
                    gf.write("Grad in-> {}\n".format(gradin))
                    gf.write("Grad out 0 -> {}\n".format(gradout))
            if epoch > 4 and epoch < 7:
                model.FC_layer[4].register_backward_hook(loss_hook)
            """
            target = target.view(-1)
            m_f, margin_pred, ori_pred = model(points, target) 
            # loss_softmax = criterion(pred, target)
            loss_softmax = criterion(margin_pred, target)
            #loss_cent = criterion_cent(m_features, target)
            loss = loss_softmax
            loss.backward()
            optimizer.step()

            # by doing so, weight_cent would not impact on the learning of centers
            #for param in criterion_cent.parameters():
            #    param.grad.data *= (1. / 0.05)
            #optimizer_cent.step()

            if i % args.print_freq_iter == 0:
                print('[epoch %3d: %3d/%3d] \t train loss: %0.6f \t lr: %0.5f' %(epoch+1, i, num_batch, loss.item(), lr_scheduler.get_lr()[0]))
            batch_count += 1
            
            # validation in between an epoch
            if args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
                validate(test_dataloader, model, criterion, args, batch_count, epoch)
        

def validate(test_dataloader, model, criterion, args, iter, epoch): 
    global g_loss
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for j, data in enumerate(test_dataloader, 0):
            points, target = data
            points, target = points.cuda(), target.cuda()
            points, target = Variable(points), Variable(target)
            
            # fastest point sampling
            fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
            # fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

            # initialize augmentation
            # points = data_augmentation(points)

            target = target.view(-1)
            m_f, _, pred = model(points, target)  
            loss_softmax = criterion(pred, target)
            # l2_loss = glob_criterion(m_f, target)
            loss = loss_softmax
            losses.append(loss.item())
            _, pred_choice = torch.max(pred, -1)

            
            preds.append(pred_choice)
            labels.append(target.data)
            
        preds = torch.cat(preds, 0)
        labels = torch.cat(labels, 0)
        #print(torch.sum(preds == labels), labels.numel())
        acc = torch.sum(preds == labels).item()/labels.numel()
        avg_loss = np.array(losses).mean()
        print('\nval loss: %0.6f \t acc: %0.6f\n'%(avg_loss, acc))
        
        if avg_loss <= g_loss:
            if avg_loss > 6:
                g_loss = avg_loss
            torch.save(model.state_dict(), '%s/face_id_ssn_iter_%d_loss_%0.6f_4SA_DA.pth' % (args.save_path, iter, avg_loss))
            # compute accurarcy of each category
            """tmp_preds = preds.cpu().numpy()
            tmp_labels = labels.cpu().numpy()
            total_cate = np.zeros(labels.numel(), dtype=np.int)
            acc_cate = np.zeros(labels.numel(), dtype=np.int)
            for i in np.arange(labels.numel()):
                total_cate[tmp_labels[i]] += 1
                if tmp_preds[i] == tmp_labels[i]:
                    acc_cate[tmp_labels[i]] +=1
            with open('log/face_expression_acc_10.txt', 'w') as m_f:
                m_f.write('\t\tSU\t'+'DI\t'+'FE\t'+'AN\t'+'SA\t'+'NE\t'+'HA\n')
                m_f.write('acc:\t')
                for i in np.arange(tmp_labels.shape[0]):  
                    m_f.write(str(int(acc_cate[i]))+'\t')
                m_f.write('\nall:\t')
                for i in np.arange(tmp_labels.shape[0]):  
                    m_f.write(str(int(total_cate[i]))+'\t')
            """

    model.train()
    
if __name__ == "__main__":
    main()