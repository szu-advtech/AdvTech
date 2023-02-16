import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not BASE_DIR in sys.path:
    sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils.pytorch_utils as pt_utils
from pct import Point_Transformer_Last, Pct
from utils.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
from AMS import AngleLinear
import numpy as np

class complementary_learning(nn.Module):
    def __init__(self, in_channel=512):
        super(complementary_learning, self).__init__()
        # self.fc1 = nn.Linear(in_channel, in_channel)
        # self.fc2 = nn.Linear(in_channel, in_channel)
        self.mlp1 = MLP(in_channel,2*in_channel,in_channel)
        self.mlp2 = MLP(in_channel,2*in_channel,in_channel)
        #self.softmax = gumbel_softmax

    def forward(self, input1, input2, tau=1):
        temp = torch.stack([input1,input2],dim=2)
        t = torch.zeros(temp.shape[0],temp.shape[1],1)
        f1 = self.mlp1(input1)
        f2 = self.mlp2(input2)
        input = torch.stack([f1,f2],dim=2)
        # soft_out = F.softmax(input,dim=2)
        # out = temp*soft_out
        # out = out[:,:,0] + out[:,:,1]
        soft_out = F.gumbel_softmax(input, tau=tau,hard=True)
        out = temp*soft_out
        out = out[:,:,0] + out[:,:,1]
        return out

# Relation-Shape CNN: Single-Scale Neighborhood
class RSCNN_SSN_SIAMESE(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=0, relation_prior=1, use_xyz=True, local=True):
        super().__init__()

        self.need_local = local
        self.SA_modules = nn.ModuleList()

        ############### 2022.10.29 yjonben
        self.SA_modules_nonormal=nn.ModuleList()
        self.SA_modules_nonormal.append(
            PointnetSAModuleMSG(
                npoint=2048,  # 1024
                radii=[0.24],  # 0.23
                nsamples=[32],  # 48 32
                mlps=[[0, 64]],  # 128
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules_nonormal.append(
            PointnetSAModuleMSG(
                npoint=1024,  # 512
                radii=[0.28],  # 0.32 0.28
                nsamples=[48],  # 64 48
                mlps=[[64, 128]],  # 256
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules_nonormal.append(
            PointnetSAModuleMSG(
                npoint=512,  # 256
                radii=[0.32],  # 0.32
                nsamples=[64],  # 64
                mlps=[[128, 256]],  # 512
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        ###############

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=2048,  # 1024
                radii=[0.24],  # 0.23
                nsamples=[32],  # 48 32
                mlps=[[input_channels, 64]],  # 128
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,  # 512
                radii=[0.28],  # 0.32 0.28
                nsamples=[48],  # 64 48
                mlps=[[64, 128]],  # 256
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,  # 256
                radii=[0.32],  # 0.32
                nsamples=[64],  # 64
                mlps=[[128, 256]],  # 512
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        """ # No.4 SA layer"""
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,  # 128
                radii=[0.32],
                nsamples=[64],
                mlps=[[256, 512]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules.append(
            # global convolutional pooling
            PointnetSAModule(
                nsample=256,
                mlp=[512, 1024],
                use_xyz=use_xyz
            )
        )

        self.pct = Pct(args=None, output_channels=num_classes)

        self.FC_layer = nn.Sequential(
            pt_utils.FC(1024, 512, activation=nn.ReLU(inplace=True), bn=True),

            # 2022.10.26 yjonben:for concat
            # pt_utils.FC(1024, 256, activation=nn.ReLU(inplace=True), bn=True),
            #

            nn.Dropout(p=0.4),
            # pt_utils.FC(512, 512, activation=nn.ReLU(inplace=True), bn=True),
            # nn.Dropout(p=0.5),
            # pt_utils.FC(512, num_classes, activation=None)
        )  # I modified in 4/8, 2020
        # self.finalLinear = pt_utils.FC(512, num_classes, activation=None)
        # 2022.10.29 yjonben
        self.finalLinear = pt_utils.FC(768, num_classes, activation=None)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features
    

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: pair of torch.cuda.FloatTensor
                each tensor : (B, N, 3 + input_channels)
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        """
        pointcloud_A, pointcloud_B = pointcloud
        xyz_A, features_A = self._break_up_pc(pointcloud_A)
        xyz_B, features_B = self._break_up_pc(pointcloud_B)
        for module in self.SA_modules:
            xyz_A, features_A = module(xyz_A, features_A)
            xyz_B, features_B = module(xyz_B, features_B)
        mediate_features_A = self.FC_layer(features_A.squeeze(-1))
        mediate_features_B = self.FC_layer(features_B.squeeze(-1))
        pred_A = self.finalLinear(mediate_features_A)
        pred_B = self.finalLinear(mediate_features_B) 
        return mediate_features_A, pred_A, mediate_features_B, pred_B
        """

        pointcloud_A, pointcloud_B = pointcloud
        xyz_A, features_A = self._break_up_pc(pointcloud_A)
        xyz_B, features_B = self._break_up_pc(pointcloud_B)
        xyz_A_nonor=xyz_A
        xyz_B_nonor=xyz_B
        features_A_nonor=None
        features_B_nonor=None

        for i, module in enumerate(self.SA_modules):
            xyz_A, features_A = module(xyz_A, features_A)
            xyz_B, features_B = module(xyz_B, features_B)
            # if i == 2:
            #     xyz_A_Trans, features_A_Trans = xyz_A, features_A
            #     xyz_B_Trans, features_B_Trans = xyz_B, features_B

        for i, module in enumerate(self.SA_modules_nonormal):
            xyz_A_nonor, features_A_nonor = module(xyz_A_nonor, features_A_nonor)
            xyz_B_nonor, features_B_nonor = module(xyz_B_nonor, features_B_nonor)

        pred_A_Trans, mediate_features_A_Trans = self.pct(features_A_nonor)
        pred_B_Trans, mediate_features_B_Trans = self.pct(features_B_nonor)
        # print(features_A.shape)

        mediate_features_A = self.FC_layer(features_A.squeeze(-1))
        mediate_features_B = self.FC_layer(features_B.squeeze(-1))

        mediate_features_A = torch.cat([mediate_features_A, mediate_features_A_Trans],dim=1)
        mediate_features_B = torch.cat([mediate_features_B, mediate_features_B_Trans],dim=1)
        # print(mediate_features_A.shape)

        # print(mediate_features_A.shape)
        pred_A = self.finalLinear(mediate_features_A)
        pred_B = self.finalLinear(mediate_features_B)

        # pred_A = 0.5 * pred_A + 0.5 * pred_A_Trans
        # pred_B = 0.5 * pred_B + 0.5 * pred_B_Trans
        # print("pred_A: ")
        # print(pred_A.shape)

        if self.need_local == True:
            return mediate_features_A, pred_A, mediate_features_B, pred_B  # , features_local_A, features_local_B
        else:
            return mediate_features_A, mediate_features_B


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    device = torch.device("cuda:0")
    sample_data_a = torch.randn(4, 4096, 3).to(device)
    sample_data_b = torch.randn(4, 4096, 3).to(device)
    model = RSCNN_SSN_SIAMESE(1000).to(device)
    from torchsummaryX import summary

    summary(model, (sample_data_a, sample_data_b))
    # mf_a, pred_a, mf_b, pred_b, local_a, local_b = model((sample_data_a, sample_data_b))
    # print(mf_a.shape, pred_a)
