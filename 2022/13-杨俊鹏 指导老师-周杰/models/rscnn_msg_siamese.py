import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not BASE_DIR in sys.path:
    sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
import numpy as np
import pointnet2_utils

# Relation-Shape CNN: Multi-Scale Neighborhood
class RSCNN_MSG_SIAMESE(nn.Module):
    r"""
        Siamese network of PointNet2 with multi-scale features

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
        local: bool True
            Whether or not to return mediate features
    """

    def __init__(self, num_classes, input_channels=0, relation_prior=1, use_xyz=True, local=True):
        super().__init__()

        self.need_local = local
        self.SA_modules = nn.ModuleList()
        
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=2048, #2048
                radii=[0.24], #0.23
                nsamples=[32], #48 32
                mlps=[[input_channels, 64]], #128
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_0 = 64
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024, #512
                radii=[0.28], # 0.28 0.56
                nsamples=[48], #64 48
                mlps=[[c_out_0, 128]], # 256
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        c_out_1 = 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512, #512
                radii=[0.32], # 0.28 0.56
                nsamples=[64], #64 48
                mlps=[[c_out_1, 256]], # 256
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_2 = 256 

        """ # No.4 SA layer"""
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256, #128
                radii=[0.34],
                nsamples=[64],
                mlps=[[256, 512]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        
        c_out_3 = c_out_0 + c_out_1 + c_out_2 + 512

        self.SA_modules.append(
            # global convolutional pooling
            PointnetSAModule(
                nsample = 256, 
                mlp=[c_out_3, 1024], 
                use_xyz=use_xyz
            )
        )

        self.FC_layer = nn.Sequential(
            pt_utils.FC(1024, 512, activation=nn.ReLU(inplace=True), bn=True),
            nn.Dropout(p=0.4),
            # pt_utils.FC(512, 512, activation=nn.ReLU(inplace=True), bn=True),
            # nn.Dropout(p=0.5),
            #pt_utils.FC(512, num_classes, activation=None)
        )# I modified in 4/8, 2020
        self.finalLinear = pt_utils.FC(512, num_classes, activation=None)

        self.downsample_layers = nn.ModuleList()
        self.num_conv = 3
        first_dim = 64
        for i in np.arange(self.num_conv):
            self.downsample_layers.append(
                Downsmapling_layer_with_FPS(
                out_npoint=256, input_channels=first_dim, output_channels=first_dim)
            )
            first_dim *= 2

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

        # pointcloud_A, pointcloud_B = pointcloud
        # xyz_A, features_A = self._break_up_pc(pointcloud_A)
        # xyz_B, features_B = self._break_up_pc(pointcloud_B)

        # for i, module in enumerate(self.SA_modules):
        #     xyz_A, features_A = module(xyz_A, features_A)
        #     xyz_B, features_B = module(xyz_B, features_B)
        # mediate_features_A = self.FC_layer(features_A.squeeze(-1))
        # mediate_features_B = self.FC_layer(features_B.squeeze(-1))
        # pred_A = self.finalLinear(mediate_features_A)
        # pred_B = self.finalLinear(mediate_features_B) 
        # if self.need_local == True:
        #     return mediate_features_A, pred_A, mediate_features_B, pred_B
        # else:
        #     return mediate_features_A, mediate_features_B
        pointcloud_A, pointcloud_B = pointcloud
        xyz_A, features_A = self._break_up_pc(pointcloud_A)
        xyz_B, features_B = self._break_up_pc(pointcloud_B)
        #print(features_residual.shape)
        xyz_A_residual, feature_A_residual = [], []
        xyz_B_residual, feature_B_residual = [], []
        for i, module in enumerate(self.SA_modules):
            if i < len(self.SA_modules)-1:
                xyz_A, features_A = module(xyz_A, features_A)
                xyz_A_residual.append(xyz_A)
                feature_A_residual.append(features_A)
                # another point cloud
                xyz_B, features_B = module(xyz_B, features_B)
                xyz_B_residual.append(xyz_B)
                feature_B_residual.append(features_B)
        for i, module in enumerate(self.downsample_layers):
            feature_A_residual[i] = module(xyz_A_residual[i], feature_A_residual[i])
            feature_B_residual[i] = module(xyz_B_residual[i], feature_B_residual[i])

        features_A = torch.cat(feature_A_residual, dim=1)
        xyz_A, features_A = self.SA_modules[-1](xyz_A, features_A)
        features_B = torch.cat(feature_B_residual, dim=1)
        xyz_B, features_B = self.SA_modules[-1](xyz_B, features_B)

        mediate_features_A = self.FC_layer(features_A.squeeze(-1))
        mediate_features_B = self.FC_layer(features_B.squeeze(-1))

        pred_A = self.finalLinear(mediate_features_A)
        pred_B = self.finalLinear(mediate_features_B) 
        if self.need_local:
            return mediate_features_A, pred_A, mediate_features_B, pred_B
        else:
            return mediate_features_A, mediate_features_B


class Downsmapling_layer_with_FPS(nn.Module):
    def __init__(self, out_npoint=256, input_channels= 128, output_channels=256, radius=0.32, nsample=64):
        super().__init__()
        self.pointnet = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(1,1), stride=1)
        self.query_and_group = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=False)
        self.out_npoint = out_npoint
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(output_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=(1,nsample))


    def forward(self, xyz, features):
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        fps_index = pointnet2_utils.furthest_point_sample(xyz, self.out_npoint)
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, fps_index).transpose(1, 2).contiguous()
        out_feat = self.query_and_group(xyz, new_xyz, features, fps_index)
        out_feat = self.activation(self.bn(self.pointnet(out_feat)))
        out_feat = self.max_pool(out_feat)
        return out_feat.squeeze(3)


if __name__ == "__main__":
    sample_data_a = torch.randn(6, 1024, 3).cuda()
    sample_data_b = torch.randn(6, 1024, 3).cuda()
    model = RSCNN_MSG_SIAMESE(2).cuda()
    model.train()
    mf_a, pred_a, mf_b, pred_b = model((sample_data_a, sample_data_b))
    print(mf_a.shape, pred_a.shape)
    