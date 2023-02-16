import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import argparse
import random
import yaml
from tqdm import tqdm
import time

# from models import RSCNN_MS
from models import RSCNN_SSN_SIAMESE
from data import KinectLQ_eval
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Face Identification')
parser.add_argument('--config', default='cfgs/config_ssn_face_id.yaml', type=str)

NUM_REPEAT = 10
G_BATCH_SIZE = 16  # G_BATCH_SIZE * n <= total samples


def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config['common'].items():
        setattr(args, k, v)

    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    print(args.input_channels)
    model = RSCNN_SSN_SIAMESE(num_classes=args.num_classes, input_channels=args.input_channels,
                              relation_prior=args.relation_prior, use_xyz=True, local=False)

    print("you has {} GPU(s)".format(torch.cuda.device_count()))
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0])
    model.to(device)

    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))
    # test_dataset_1 = Bosphorus_eval(num_points = args.num_points+2000, root = args.data_root, transforms=test_transforms
    #     , task='id', with_noise=True)
    test_dataset_1 = KinectLQ_eval(num_points=args.num_points + 2000, root=args.data_root, transforms=test_transforms
                                   , valtxt='FE_val_du2.txt', normals=True)
    test_dataloader_1 = DataLoader(
        test_dataset_1,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers)
    )

    test_dataset_2 = KinectLQ_eval(num_points=args.num_points + 2000, root=args.data_root, transforms=test_transforms
                                   , valtxt='NU_val_du2.txt', normals=True)
    test_dataloader_2 = DataLoader(
        test_dataset_2,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers)
    )
    test_dataset_3 = KinectLQ_eval(num_points=args.num_points + 2000, root=args.data_root, transforms=test_transforms
                                   , valtxt='OC_val_du2.txt', normals=True)
    test_dataloader_3 = DataLoader(
        test_dataset_3,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers)
    )
    test_dataset_4 = KinectLQ_eval(num_points=args.num_points + 2000, root=args.data_root, transforms=test_transforms
                                   , valtxt='PS_val_du2.txt', normals=True)
    test_dataloader_4 = DataLoader(
        test_dataset_4,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers)
    )
    test_dataset_5 = KinectLQ_eval(num_points=args.num_points + 2000, root=args.data_root, transforms=test_transforms
                                   , valtxt='TM_val_du2.txt', normals=True)
    test_dataloader_5 = DataLoader(
        test_dataset_5,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers)
    )

    # model = RSCNN_SSN(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    # model = dgcnn.DGCNN(output_channels = args.num_classes)
    # model.cuda()

    # elif torch.cuda.device_count()==1:
    #   model.cuda()

    # model is used for feature extraction, so no need FC layers
    # model.finalLinear = nn.Linear(512, 512, bias=False).cuda()
    for para in model.parameters():
        para.requires_grad = False
    # nn.init.eye_(model.finalLinear.weight)

    # evaluate
    # PointcloudScale = d_utils.PointcloudScale()   # initialize random scaling
    model.eval()

    with torch.no_grad():
        Total_samples = 0
        Correct = 0
        gallery_points, gallery_labels = test_dataset_1.get_gallery()
        gallery_points, gallery_labels = gallery_points.cuda(), gallery_labels.cuda()
        gallery_points = Variable(gallery_points)

        gallery_num = gallery_labels.shape[0]
        # gallery_labels_new = torch.zeros(gallery_num//G_BATCH_SIZE, dtype=torch.long).cuda()
        for i in np.arange(0, gallery_num // G_BATCH_SIZE + 1):
            # print(gallery_points[i*G_BATCH_SIZE:i+G_BATCH_SIZE,:,:].shape)
            if i < gallery_num // G_BATCH_SIZE:
                g_points = gallery_points[i * G_BATCH_SIZE:i * G_BATCH_SIZE + G_BATCH_SIZE, :, :]
                # fps_idx = pointnet2_utils.furthest_point_sample(g_points, args.num_points)  # (B, npoint)
                # g_points = pointnet2_utils.gather_operation(g_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                tmp_pred, m_b = model((g_points, g_points))
                # tmp_pred = model(g_points)
                if i == 0:
                    gallery_pred = torch.tensor(tmp_pred).clone().cuda()
                else:
                    gallery_pred = torch.cat((gallery_pred, tmp_pred), dim=0)

            if i == gallery_num // G_BATCH_SIZE:
                num_of_rest = gallery_num % G_BATCH_SIZE
                g_points = gallery_points[i * G_BATCH_SIZE:i * G_BATCH_SIZE + num_of_rest, :, :]
                # fps_idx = pointnet2_utils.furthest_point_sample(g_points, args.num_points)  # (B, npoint)
                # g_points = pointnet2_utils.gather_operation(g_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                tmp_pred, m_b = model((g_points, g_points))
                # tmp_pred = model(g_points)
                gallery_pred = torch.cat((gallery_pred, tmp_pred), dim=0)
                # print(tmp_pred.size(), gallery_pred.size())

        # gallery_labels_new = gallery_labels.clone()
        print("gallery features size:{}".format(gallery_pred.size()))
        gallery_pred = F.normalize(gallery_pred)
        # print("labels: {}".format(gallery_labels_new.data))

        """Eval dataset 1"""
        for j, data in enumerate(tqdm(test_dataloader_1)):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda()
            probe_points = Variable(probe_points)

            # get feature vetor for probe and gallery set from model
            # fps_idx = pointnet2_utils.furthest_point_sample(probe_points, args.num_points)  # (B, npoint)
            # probe_points = pointnet2_utils.gather_operation(probe_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            start_time = time.time()
            probe_pred, m_b = model((probe_points, probe_points))
            end_time = time.time()
            # probe_pred = model(probe_points)
            probe_pred = F.normalize(probe_pred)

            # make tensor to size (probe_num, gallery_num, C)
            probe_tmp = probe_pred.unsqueeze(1).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                       probe_pred.shape[1])
            gallery_tmp = gallery_pred.unsqueeze(0).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                           gallery_pred.shape[1])
            results = torch.sum(torch.mul(probe_tmp, gallery_tmp), dim=2)  # cosine distance
            # results = torch.sum(torch.pow(probe_tmp-gallery_tmp, 2), dim=2) # l2 distance
            results = torch.argmax(results, dim=1)
            # results = torch.argmin(results, dim=1) # l2 judge

            Total_samples += probe_points.shape[0]
            # print(results, probe_labels)
            for i in np.arange(0, results.shape[0]):
                if gallery_labels[results[i]] == probe_labels[i]:
                    Correct += 1
        # print('Eval Set 1 total_samples:{}'.format(Total_samples))
        # acc = float(Correct/Total_samples)
        # print('\n test_dataset_1 {} acc: {:.6f}\n'.format(test_dataset_1.valtxt, acc))

        """Eval dataset 2"""
        # Total_samples = 0
        # Correct = 0
        for j, data in enumerate(tqdm(test_dataloader_2)):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda()
            probe_points = Variable(probe_points)

            # get feature vetor for probe and gallery set from model
            # fps_idx = pointnet2_utils.furthest_point_sample(probe_points, args.num_points)  # (B, npoint)
            # probe_points = pointnet2_utils.gather_operation(probe_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            probe_pred, m_b = model((probe_points, probe_points))
            # probe_pred = model(probe_points)
            probe_pred = F.normalize(probe_pred)

            # make tensor to size (probe_num, gallery_num, C)
            probe_tmp = probe_pred.unsqueeze(1).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                       probe_pred.shape[1])
            gallery_tmp = gallery_pred.unsqueeze(0).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                           gallery_pred.shape[1])
            results = torch.sum(torch.mul(probe_tmp, gallery_tmp), dim=2)  # cosine distance
            # results = torch.sum(torch.pow(probe_tmp-gallery_tmp, 2), dim=2) # l2 distance
            results = torch.argmax(results, dim=1)
            # results = torch.argmin(results, dim=1) # l2 judge

            Total_samples += probe_points.shape[0]
            # print(results, probe_labels)
            for i in np.arange(0, results.shape[0]):
                if gallery_labels[results[i]] == probe_labels[i]:
                    Correct += 1
        for j, data in enumerate(tqdm(test_dataloader_3)):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda()
            probe_points = Variable(probe_points)

            # get feature vetor for probe and gallery set from model
            # fps_idx = pointnet2_utils.furthest_point_sample(probe_points, args.num_points)  # (B, npoint)
            # probe_points = pointnet2_utils.gather_operation(probe_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            probe_pred, m_b = model((probe_points, probe_points))
            # probe_pred = model(probe_points)
            probe_pred = F.normalize(probe_pred)

            # make tensor to size (probe_num, gallery_num, C)
            probe_tmp = probe_pred.unsqueeze(1).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                       probe_pred.shape[1])
            gallery_tmp = gallery_pred.unsqueeze(0).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                           gallery_pred.shape[1])
            results = torch.sum(torch.mul(probe_tmp, gallery_tmp), dim=2)  # cosine distance
            # results = torch.sum(torch.pow(probe_tmp-gallery_tmp, 2), dim=2) # l2 distance
            results = torch.argmax(results, dim=1)
            # results = torch.argmin(results, dim=1) # l2 judge

            Total_samples += probe_points.shape[0]
            # print(results, probe_labels)
            for i in np.arange(0, results.shape[0]):
                if gallery_labels[results[i]] == probe_labels[i]:
                    Correct += 1
        for j, data in enumerate(tqdm(test_dataloader_4)):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda()
            probe_points = Variable(probe_points)

            # get feature vetor for probe and gallery set from model
            # fps_idx = pointnet2_utils.furthest_point_sample(probe_points, args.num_points)  # (B, npoint)
            # probe_points = pointnet2_utils.gather_operation(probe_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            probe_pred, m_b = model((probe_points, probe_points))
            # probe_pred = model(probe_points)
            probe_pred = F.normalize(probe_pred)

            # make tensor to size (probe_num, gallery_num, C)
            probe_tmp = probe_pred.unsqueeze(1).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                       probe_pred.shape[1])
            gallery_tmp = gallery_pred.unsqueeze(0).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                           gallery_pred.shape[1])
            results = torch.sum(torch.mul(probe_tmp, gallery_tmp), dim=2)  # cosine distance
            # results = torch.sum(torch.pow(probe_tmp-gallery_tmp, 2), dim=2) # l2 distance
            results = torch.argmax(results, dim=1)
            # results = torch.argmin(results, dim=1) # l2 judge

            Total_samples += probe_points.shape[0]
            # print(results, probe_labels)
            for i in np.arange(0, results.shape[0]):
                if gallery_labels[results[i]] == probe_labels[i]:
                    Correct += 1
        for j, data in enumerate(tqdm(test_dataloader_5)):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda()
            probe_points = Variable(probe_points)

            # get feature vetor for probe and gallery set from model
            # fps_idx = pointnet2_utils.furthest_point_sample(probe_points, args.num_points)  # (B, npoint)
            # probe_points = pointnet2_utils.gather_operation(probe_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            probe_pred, m_b = model((probe_points, probe_points))
            # probe_pred = model(probe_points)
            probe_pred = F.normalize(probe_pred)

            # make tensor to size (probe_num, gallery_num, C)
            probe_tmp = probe_pred.unsqueeze(1).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                       probe_pred.shape[1])
            gallery_tmp = gallery_pred.unsqueeze(0).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                           gallery_pred.shape[1])
            results = torch.sum(torch.mul(probe_tmp, gallery_tmp), dim=2)  # cosine distance
            # results = torch.sum(torch.pow(probe_tmp-gallery_tmp, 2), dim=2) # l2 distance
            results = torch.argmax(results, dim=1)
            # results = torch.argmin(results, dim=1) # l2 judge

            Total_samples += probe_points.shape[0]
            # print(results, probe_labels)
            for i in np.arange(0, results.shape[0]):
                if gallery_labels[results[i]] == probe_labels[i]:
                    Correct += 1
        print('Eval Set 2 total_samples:{}'.format(Total_samples))
        acc = float(Correct / Total_samples)
        print('\n total acc: {:.6f}\n'.format(acc))
        # one_batch_infer_time = (end_time-start_time)*1000 # ms
        # print("inference time: {:.6f}".format(one_batch_infer_time/args.batch_size))


if __name__ == '__main__':
    main()