import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
import AMS
import numpy as np

# Relation-Shape CNN: Single-Scale Neighborhood
class RSCNN_SSN_MarginSoftmax(nn.Module):
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

    def __init__(self, num_classes, input_channels=0, relation_prior=1, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024, #1024
                radii=[0.23], #0.23 0.24
                nsamples=[32], #48
                mlps=[[input_channels, 128]], #128
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512, #512
                radii=[0.28], #0.32 0.28
                nsamples=[48],
                mlps=[[128, 256]], #128,256
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        # modified in 2020/5/16
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256, #256
                radii=[0.32],
                nsamples=[64],
                mlps=[[256, 512]], #256 512
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        
        
        self.SA_modules.append(
            # global convolutional pooling
            PointnetSAModule(
                nsample=256, #128
                mlp=[512, 1024], 
                use_xyz=use_xyz
            )
        )

        self.FC_layer = nn.Sequential(
            pt_utils.FC(1024, 512, activation=nn.ReLU(inplace=True), bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(512, 512, activation=nn.ReLU(inplace=True), bn=True),
            nn.Dropout(p=0.5),
        #     #pt_utils.FC(512, num_classes, activation=None)
        )# I modified in 4/8, 2020
        self.finalLinear =  AMS.ArcFace(512, num_classes, m=0.35,s=8) #pt_utils.FC(512, num_classes, activation=None)
        

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, target=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        mediate_features = self.FC_layer(features.squeeze(-1))
        #mediate_features = features.squeeze(-1)
        if target is None:
            return mediate_features
        else:
            pred_margin, pred = self.finalLinear(mediate_features, target)
            return mediate_features, pred_margin, pred


if __name__ == "__main__":
    sim_data = torch.rand(8, 1024, 3)
    sim_data = sim_data.cuda()
    sim_cls = torch.tensor([0,2,5,6,1,1,3,4], dtype=torch.long)
    sim_cls = sim_cls.cuda()

    model = RSCNN_SSN_MarginSoftmax(num_classes=8, input_channels=0, use_xyz=True)
    model = model.cuda()
    mf, out, out_margin = model(sim_data, target=sim_cls)
    print('out size: {}'.format(out_margin.size()))
    print('embeding size: {}'.format(mf.size()))