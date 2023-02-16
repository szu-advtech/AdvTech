import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
from torchvision import transforms
from models import RSCNN_SSN_SIAMESE, RSCNN_MSG_SIAMESE
from data import KinectLQ_train_pair, FRGC_train_pair, Bosphorus_eval
import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
from models.ContrastiveLoss import MultiContrastiveLoss
import argparse
import random
import yaml
import math
#import visdom
from tqdm import tqdm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 

parser = argparse.ArgumentParser(description='Relation-Shape CNN Face Classification Training(Siamese Network)')
parser.add_argument('--config', default='cfgs/config_ssn_face_id.yaml', type=str)

def data_augmentation(point_cloud):
    """point cloud data augmentation using function from data_utils.py"""

    PointcloudRandomInputDropout = d_utils.PointcloudRandomInputDropout()
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()
    PointcloudJitter = d_utils.PointcloudJitter(clip=0.02)
    # PointcloudJitter = d_utils.PointcloudJitterAxisZ(std=0.05)
    # PointcloudOcclusion = d_utils.PointcloudManmadeOcclusion()
    angle = math.pi/2.0
    
    PointcloudRotatebyRandomAngle_y = d_utils.PointcloudRotatebyRandomAngle(rotation_angle=angle, 
        axis=np.array([0.0, 1.0, 0.0]))
    PointcloudRotatebyRandomAngle_x = d_utils.PointcloudRotatebyRandomAngle(rotation_angle=angle, 
        axis=np.array([1.0, 0.0, 0.0]))
    PointcloudRotatebyRandomAngle_z = d_utils.PointcloudRotatebyRandomAngle(rotation_angle=angle, 
        axis=np.array([0.0, 0.0, 1.0]))  
  

    transform_func = {0: lambda x: x,
            1:PointcloudScaleAndTranslate,
            2:PointcloudRotatebyRandomAngle_x,
            3:PointcloudRotatebyRandomAngle_y,
            # 4:PointcloudRotatebyRandomAngle_z,
            # 4:PointcloudOcclusion
            4:PointcloudJitter,
            }

    method_num = 1

    func_id = np.array(list(transform_func.keys()))
    pro = [0.2, 0.2, 0.2, 0.2, 0.2] # probability of each transform function
    # other_pro = (1-pro[0])/float(len(transform_func)-1)
    # pro.extend([other_pro]*(func_id.shape[0]-1))

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
    
    train_dataset = KinectLQ_train_pair(num_points=args.num_points+2000, root=args.data_root, transforms=train_transforms, train=True, normals=True)
    # train_dataset = FRGC_train_pair(num_points=args.num_points+2000, root=args.data_root, transforms=train_transforms, train=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers = int(args.workers),
        drop_last=False
    )

    # test_dataset = Bosphorus_eval(num_points=args.num_points+2000, root=args.data_root, transforms=test_transforms, with_noise=True, task='id')
    test_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers)
    )
    
    model = RSCNN_SSN_SIAMESE(num_classes=args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    # model = RSCNN_MSG_SIAMESE(num_classes=args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>=2:
        model = nn.DataParallel(model, device_ids=[0,1])
        model.to(device)
    elif torch.cuda.device_count()==1:
        model.cuda()

    optimizer = optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.weight_decay)

    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)
    
    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))

    criterion = nn.CrossEntropyLoss()
    #criterion_cos = nn.CosineEmbeddingLoss(margin=0.1)
    criterion_globfeat = MultiContrastiveLoss(margin=0.35, scale=1.0) # L2 -> margin=1.6, scale=1; Cosine->0.35 0.4finetune
    #criterion_localfeat = IntermediateLoss(margin=0.3)

    num_batch = len(train_dataset)/args.batch_size
    
    # training
    # train(train_dataloader, test_dataset, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch, criterion_globfeat)# for frgcv2 and bos
    train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch, criterion_globfeat)# for lock3dface
    

def train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch, criterion_globfeat):
    global sum_loss 
    sum_loss = 8.2   # only save the model whose loss < 2.0
    batch_count = 0
    model.train()
    for epoch in range(args.epochs): # from epoch 69 because of shutdown
        losses = []
        for i, data in enumerate(train_dataloader, 0):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
              bnm_scheduler.step(epoch-1)
            points1, target1, points2, target2 = data
            points1, target1, points2, target2 = points1.cuda(), target1.cuda(), points2.cuda(), target2.cuda()
            points1, target1 = Variable(points1), Variable(target1)
            points2, target2 = Variable(points2), Variable(target2)
            
            # fastest point sampling
            fps_idx = pointnet2_utils.furthest_point_sample(points1, args.num_points)  # (B, npoint)
            points1 = pointnet2_utils.gather_operation(points1.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            fps_idx = pointnet2_utils.furthest_point_sample(points2, args.num_points)  # (B, npoint)
            points2 = pointnet2_utils.gather_operation(points2.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)            
            # augmentation
            points1 = data_augmentation(points1)
            points2 = data_augmentation(points2)
            
            optimizer.zero_grad()
            
            m_features_A, pred_A, m_features_B, pred_B = model((points1, points2))
            #m_features_A, pred_A, m_features_B, pred_B, locfeat_A, locfeat_B = model((points1, points2))
            target1, target2 = target1.view(-1), target2.view(-1)
            loss_A = criterion(pred_A, target1)
            loss_B = criterion(pred_B, target2)

            # need to compute L2 loss between 2 mediate features
            # cos_target = torch.zeros(target1.shape, dtype=torch.float).cuda()
            # cos_target[target1==target2] = 1
            # cos_target[target1!=target2] = 0

            loss_gf = criterion_globfeat(m_features_A, m_features_B, target1, target2)
            #loss_lf = criterion_localfeat(locfeat_A, locfeat_B, cos_target)
            
            loss = loss_A + loss_B +  loss_gf
            loss.backward()
            optimizer.step()

            if i % args.print_freq_iter == 0:
                print('[epoch %3d: %3d/%3d] \t train loss: %0.6f \t lr: %0.5f' %(epoch+1, i, num_batch, loss.item(), lr_scheduler.get_lr()[0]))
            batch_count += 1
            
            losses.append(loss.item())
            # validation in between an epoch
            # if args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
            #     validate(test_dataset, test_dataloader, model, criterion, args, batch_count, criterion_globfeat, epoch)
        avg_loss = np.array(losses).mean()
        print('\nval loss: {:.6f}'.format(avg_loss))
        if epoch >=60 and avg_loss <= sum_loss:
        # if avg_loss <= sum_loss:
            if avg_loss >= 2.0:
                sum_loss = avg_loss
            torch.save(model.state_dict(), '%s/cls_ssn_iter_%d_loss_%0.6f_PositiveMulti_Cos0.30.pth' % (args.save_path, batch_count, avg_loss))


def validate(test_dataset, test_dataloader, model, criterion, args, iter, criterion_globfeat, epoch): 
    global sum_loss

    model.eval()
    losses= []
    preds_A, labels_A, preds_B, labels_B = [], [], [], [] 
    with torch.no_grad():
        for j, data in enumerate(test_dataloader, 0):
            points1, target1, points2, target2 = data
            points1, target1, points2, target2 = points1.cuda(), target1.cuda(), points2.cuda(), target2.cuda()
            points1, target1 = Variable(points1), Variable(target1)
            points2, target2 = Variable(points2), Variable(target2)
            
            # fastest point sampling
            fps_idx = pointnet2_utils.furthest_point_sample(points1, args.num_points)  # (B, npoint)
            points1 = pointnet2_utils.gather_operation(points1.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            fps_idx = pointnet2_utils.furthest_point_sample(points2, args.num_points)  # (B, npoint)
            points2 = pointnet2_utils.gather_operation(points2.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)            
            
            # augmentation
            # points1 = data_augmentation(points1)
            # points2 = data_augmentation(points2)
            
            m_features_A, pred_A, m_features_B, pred_B = model((points1, points2))
            target1, target2 = target1.view(-1), target2.view(-1)
            loss_A = criterion(pred_A, target1)
            loss_B = criterion(pred_B, target2)
            # need to compute cosine loss between 2 mediate features
            # cos_target = torch.zeros(target1.shape, dtype=torch.float).cuda()
            # cos_target[target1==target2] = 1
            # cos_target[target1!=target2] = 0
            
            loss_gf = criterion_globfeat(m_features_A, m_features_B, target1, target2)
            #loss_lf = criterion_localfeat(locfeat_A, locfeat_B, cos_target)
            #loss = 0.5*(loss_A + loss_B) + loss_gf + loss_lf
            loss = loss_A + loss_B + loss_gf

            losses.append(loss.item())
            _, pred_choice_A = torch.max(pred_A, -1)
            _, pred_choice_B = torch.max(pred_B, -1)
            
            preds_A.append(pred_choice_A)
            preds_B.append(pred_choice_B)
            labels_A.append(target1)
            labels_B.append(target2)
            
        preds_A, preds_B = torch.cat(preds_A, 0), torch.cat(preds_B, 0)
        labels_A, labels_B = torch.cat(labels_A, 0), torch.cat(labels_B, 0)
        #print(torch.sum(preds == labels), labels.numel())
        acc_A = torch.sum(preds_A == labels_A).item()/labels_A.numel()
        acc_B = torch.sum(preds_B == labels_B).item()/labels_B.numel()
        avg_loss = np.array(losses).mean()
        print('\nval loss: %0.6f \t acc pair: %0.6f %0.6f\n'%(avg_loss, acc_A, acc_B))
    # model.eval()
    # with torch.no_grad():
    #     G_BATCH_SIZE=16
    #     Total_samples = 0
    #     Correct = 0
    #     gallery_points, gallery_labels = test_dataset.get_gallery()
    #     gallery_points, gallery_labels = gallery_points.cuda(), gallery_labels.cuda()
    #     gallery_points =  Variable(gallery_points)

    #     gallery_num = gallery_labels.shape[0]
    #     #gallery_labels_new = torch.zeros(gallery_num//G_BATCH_SIZE, dtype=torch.long).cuda()
    #     for i in np.arange(0, gallery_num//G_BATCH_SIZE + 1):
    #         #print(gallery_points[i*G_BATCH_SIZE:i+G_BATCH_SIZE,:,:].shape)
    #         if i < gallery_num//G_BATCH_SIZE:
    #             g_points = gallery_points[i*G_BATCH_SIZE:i*G_BATCH_SIZE+G_BATCH_SIZE,:,:]
    #             fps_idx = pointnet2_utils.furthest_point_sample(g_points, args.num_points)  # (B, npoint)
    #             g_points = pointnet2_utils.gather_operation(g_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
    #             tmp_pred, p_a, m_b, p_b = model((g_points, g_points))
    #             # tmp_pred = model(gallery_points[i*G_BATCH_SIZE:i*G_BATCH_SIZE+G_BATCH_SIZE,:,:])
    #             if i==0:
    #                 gallery_pred = torch.tensor(tmp_pred).clone().cuda()
    #             else:
    #                 gallery_pred = torch.cat((gallery_pred, tmp_pred), dim=0)

    #         if i==gallery_num//G_BATCH_SIZE:
    #             num_of_rest = gallery_num % G_BATCH_SIZE
    #             g_points = gallery_points[i*G_BATCH_SIZE:i*G_BATCH_SIZE+num_of_rest,:,:]
    #             fps_idx = pointnet2_utils.furthest_point_sample(g_points, args.num_points)  # (B, npoint)
    #             g_points = pointnet2_utils.gather_operation(g_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
    #             tmp_pred, p_a, m_b, p_b = model((g_points, g_points))
                
    #             gallery_pred = torch.cat((gallery_pred, tmp_pred), dim=0)
    #             # print(tmp_pred.size(), gallery_pred.size())

    #     # gallery_labels_new = gallery_labels.clone()
    #     # print("gallery features size:{}".format(gallery_pred.size()))
    #     gallery_pred = F.normalize(gallery_pred)
    #     # print("labels: {}".format(gallery_labels_new.data))

    #     """Eval dataset 1"""
    #     for j, data in enumerate(tqdm(test_dataloader)):
    #         probe_points, probe_labels = data
    #         probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda()           
    #         fps_idx = pointnet2_utils.furthest_point_sample(probe_points, args.num_points)  # (B, npoint)
    #         probe_points = pointnet2_utils.gather_operation(probe_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            
    #         # get feature vetor for probe and gallery set from model
    #         probe_pred, p_a, m_b, p_b = model((probe_points, probe_points))
                      
    #         probe_pred = F.normalize(probe_pred)
           
    #         # make tensor to size (probe_num, gallery_num, C)
    #         probe_tmp = probe_pred.unsqueeze(1).expand(probe_pred.shape[0], gallery_pred.shape[0], 
    #                                                     probe_pred.shape[1])
    #         gallery_tmp = gallery_pred.unsqueeze(0).expand(probe_pred.shape[0], gallery_pred.shape[0],
    #                                                     gallery_pred.shape[1])
    #         results = torch.sum(torch.mul(probe_tmp, gallery_tmp), dim=2) # cosine distance
    #         # results = torch.sum(torch.pow(probe_tmp-gallery_tmp, 2), dim=2) # l2 distance
    #         results = torch.argmax(results, dim=1)
    #         # results = torch.argmin(results, dim=1) # l2 judge
            
    #         Total_samples += probe_points.shape[0]
    #         #print(results, probe_labels)
    #         for i in np.arange(0, results.shape[0]):
    #             if gallery_labels[results[i]] == probe_labels[i]:
    #                 Correct += 1
    #     print('Eval Set 1 total_samples:{}'.format(Total_samples))      
    #     acc = float(Correct/Total_samples)
    #     print('\n test_dataset {} acc: {:.6f}\n'.format(test_dataset.__class__.__name__, acc))
        
        if avg_loss <= sum_loss:
            if avg_loss >= 3.0:
                sum_loss = avg_loss
            torch.save(model.state_dict(), '%s/cls_ssn_iter_%d_loss_%0.6f_PositiveMulti_Cos0.4_.pth' % (args.save_path, iter, avg_loss))
        # if acc > 0.56 :
        #     torch.save(model.state_dict(), '%s/cls_ssn_iter_%d_acc_%0.6f_PositiveMulti_Cos0.35_.pth' % (args.save_path, iter, acc))
            # compute accurarcy of each category
            """
            tmp_preds = preds.cpu().numpy()
            tmp_labels = labels.cpu().numpy()
            
            total_cate = np.zeros(7, dtype=np.int)
            acc_cate = np.zeros(7, dtype=np.int)
            for i in np.arange(tmp_labels.shape[0]):
                total_cate[tmp_labels[i]] += 1
                if tmp_preds[i] == tmp_labels[i]:
                    acc_cate[tmp_labels[i]] +=1
            with open('log/face_expression_Bos/face_expression_acc_0.txt', 'w') as m_f:
                m_f.write(str(tmp_labels.shape[0])+'\n')
                m_f.write('\t\tSU\t'+'DI\t'+'FE\t'+'AN\t'+'SA\t'+'NE\t'+'HA\n')
                m_f.write('acc:\t')
                for i in np.arange(acc_cate.shape[0]):  
                    m_f.write(str(int(acc_cate[i]))+'\t')
                m_f.write('\nall:\t')
                for i in np.arange(total_cate.shape[0]):  
                    m_f.write(str(int(total_cate[i]))+'\t')
            """


    model.train()
    
if __name__ == "__main__":
    main()