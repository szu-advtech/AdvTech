import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms
from models import RSCNN_SSN_Cls as RSCNN_SSN
from models import RSCNN_SSN_SIAMESE
from data import Bosphorus_eval, BU3DFE_eval, FRGC_eval, KinectLQ_eval
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Face Verification')
parser.add_argument('--config', default='cfgs/config_ssn_face_id.yaml', type=str)

NUM_REPEAT = 10
G_BATCH_SIZE=16 # G_BATCH_SIZE * n <= total samples
THRESHOLD = 0.83 # 0.79 FAR is 0.001

def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config['common'].items():
        setattr(args, k, v)
    
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    #test_dataset = BU3DFE_eval(num_points = args.num_points, root = args.data_root, transforms=test_transforms)
    test_dataset_1 = Bosphorus_eval(num_points = args.num_points+2000, root = args.data_root, transforms=test_transforms,
        with_noise=True, normal=True)
    #test_dataset = FRGC_eval(num_points = args.num_points, root = args.data_root, transforms=test_transforms)
    test_dataloader_1 = DataLoader(
        test_dataset_1, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers)
    )

    
    model = RSCNN_SSN_SIAMESE(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True, local=False)
    #model = RSCNN_SSN(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    #model = dgcnn.DGCNN(output_channels = args.num_classes)
    #model.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>=2:
        model = nn.DataParallel(model, device_ids=[0])
        model.to(device)
    elif torch.cuda.device_count()==1:
        model.cuda()

    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))
    
    # model is used for feature extraction, so no need FC layers
    #model.finalLinear = nn.Linear(512, 512, bias=False).cuda()
    for para in model.parameters():
        para.requires_grad = False
    #nn.init.eye_(model.finalLinear.weight)

    # evaluate
    #PointcloudScale = d_utils.PointcloudScale()   # initialize random scaling
    model.eval()

    with torch.no_grad():
        # TAR FAR for probe-gallery pair data
        total_pos_pair = 0
        true_accept_pair = 0
        total_neg_pair = 0
        false_accept_pair = 0
        # TPR FPR for  probe-gallery pair data
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0

        gallery_points, gallery_labels = test_dataset_1.get_gallery()
        gallery_points, gallery_labels = gallery_points.cuda(), gallery_labels.cuda()
        gallery_points =  Variable(gallery_points)

        gallery_num = gallery_labels.shape[0]
        #gallery_labels_new = torch.zeros(gallery_num//G_BATCH_SIZE, dtype=torch.long).cuda()
        for i in np.arange(0, gallery_num//G_BATCH_SIZE + 1):
            #print(gallery_points[i*G_BATCH_SIZE:i+G_BATCH_SIZE,:,:].shape)
            if i < gallery_num//G_BATCH_SIZE:
                g_points = gallery_points[i*G_BATCH_SIZE:i*G_BATCH_SIZE+G_BATCH_SIZE,:,:]
                tmp_pred, m_b = model((g_points, g_points))
                # tmp_pred = model(gallery_points[i*G_BATCH_SIZE:i*G_BATCH_SIZE+G_BATCH_SIZE,:,:])
                if i==0:
                    gallery_pred = torch.tensor(tmp_pred).clone().cuda()
                else:
                    gallery_pred = torch.cat((gallery_pred, tmp_pred), dim=0)

            if i==gallery_num//G_BATCH_SIZE:
                num_of_rest = gallery_num % G_BATCH_SIZE
                g_points = gallery_points[i*G_BATCH_SIZE:i*G_BATCH_SIZE+num_of_rest,:,:]
                tmp_pred, m_b = model((g_points, g_points))
                # tmp_pred = model(gallery_points[i*G_BATCH_SIZE:i*G_BATCH_SIZE+G_BATCH_SIZE,:,:])
                gallery_pred = torch.cat((gallery_pred, tmp_pred), dim=0)
                print(tmp_pred.size(), gallery_pred.size())

        print("gallery features size:{}".format(gallery_pred.size()))
        gallery_pred = F.normalize(gallery_pred)
        print("labels: {}".format(gallery_labels.data))

        """Eval dataset 1"""
        for j, data in enumerate(test_dataloader_1, 0):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda()           
            probe_points = Variable(probe_points)
            
            # get feature vetor for probe and gallery set from model
            probe_pred, m_b = model((probe_points, probe_points))
            # probe_pred = model(probe_points)           
            probe_pred = F.normalize(probe_pred)
           
            # make tensor to size (probe_num, gallery_num, C)
            probe_tmp = probe_pred.unsqueeze(1).expand(probe_pred.shape[0], gallery_pred.shape[0], 
                                                        probe_pred.shape[1])
            gallery_tmp = gallery_pred.unsqueeze(0).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                        gallery_pred.shape[1])
            results = torch.sum(torch.mul(probe_tmp, gallery_tmp), dim=2) # cosine distance
            # results = torch.argmax(results, dim=1)
            for i in np.arange(0, results.shape[0]):
                if results[i][gallery_labels == probe_labels[i]] >= THRESHOLD:
                    true_accept_pair += 1
                    true_pos += 1
                else:
                    false_neg += 1  
                neg_score_arr = results[i][gallery_labels != probe_labels[i]]
                false_accept_pair += torch.sum(neg_score_arr>THRESHOLD).item()
                # false_pos += torch.sum(neg_score_arr>=THRESHOLD).item()
                # true_neg += torch.sum(neg_score_arr<THRESHOLD).item()
            total_pos_pair += probe_points.shape[0]
            total_neg_pair += probe_points.shape[0] * (gallery_labels.shape[0]-1)

        print('Eval Set 1 total_pos_pair:{}'.format(total_pos_pair))
        print('Eval Set 1 total_neg_pair:{}'.format(total_neg_pair))      
        false_accept_rate = float(false_accept_pair) / float(total_neg_pair)
        true_accept_rate = float(true_accept_pair) / float(total_pos_pair)
        print('\n test_dataset_1 {} TAR: {:.6f}\n'.format(test_dataset_1.__class__.__name__, true_accept_rate))
        print('\n test_dataset_1 {} FAR: {:.6f}\n'.format(test_dataset_1.__class__.__name__, false_accept_rate))

        # print('Eval Set 1 True Positive:{}'.format(true_pos))
        # print('Eval Set 1 False Positive:{}'.format(false_pos))
        # print('Eval Set 1 True Negative:{}'.format(true_neg))
        # print('Eval Set 1 False Negative:{}'.format(false_neg))


if __name__ == '__main__':
    main()