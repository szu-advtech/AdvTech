import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
import pointnet2_utils
#from models import AMS
import numpy as np
import os

# Relation-Shape CNN: Single-Scale Neighborhood, Multi-Scale feature
class RSCNN_MS(nn.Module):
    r"""
        RSCNN with multi-scale feature
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

        self.num_conv = 2
        self.downsample_layers = nn.ModuleList()
        first_dim = 4*32
        for i in np.arange(self.num_conv):
            """if i==0:
                downsample_layers.append(Downsmapling_layer_with_FPS(
                    out_npoint=256, input_channels=input_channels+3, output_channels=512))
            else:"""    
            self.downsample_layers.append(Downsmapling_layer_with_FPS(
                out_npoint=256, input_channels=first_dim, output_channels=256))
            first_dim = first_dim*2

        self.SA_modules = nn.ModuleList()
        
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024, #512
                radii=[0.23],
                nsamples=[48],
                mlps=[[input_channels, 128, 128]],
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512, #128
                radii=[0.32],
                nsamples=[64],
                mlps=[[128, 256, 256]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        # modified in 2020/5/16
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256, #128
                radii=[0.32],
                nsamples=[64],
                mlps=[[256, 256]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        
        self.SA_modules.append(
            # global convolutional pooling
            # output size (B, mlp[-1], 1)
            PointnetSAModule(
                nsample=256, #128
                mlp=[256*(self.num_conv+1), 1024], 
                use_xyz=use_xyz
            )
        )

        self.FC_layer = nn.Sequential(
            pt_utils.FC(1024, 512, activation=nn.ReLU(), bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(512, 512, activation=nn.ReLU(), bn=True),
            nn.Dropout(p=0.5),
            #pt_utils.FC(512, num_classes, activation=None)
        )# I modified in 4/8, 2020
        #self.finalLinear =  AMS.AngleLinear(512, num_classes, margin=0.35) #pt_utils.FC(512, num_classes, activation=None)
        self.finalLinear = pt_utils.FC(512, num_classes, activation=None)

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
        #print(features_residual.shape)
        xyz_residual, feature_residual = [], []
        for i, module in enumerate(self.SA_modules):
            if i < len(self.SA_modules)-1:
                xyz, features = module(xyz, features)
                xyz_residual.append(xyz)
                feature_residual.append(features)
        for i, module in enumerate(self.downsample_layers):
            feature_residual[i] = module(xyz_residual[i], feature_residual[i])

        features = torch.cat(feature_residual, dim=1)
        xyz, features = self.SA_modules[-1](xyz, features)

        mediate_features = self.FC_layer(features.squeeze(-1))
        if target is None:
            return mediate_features
        else:
            return mediate_features, self.finalLinear(mediate_features)

class Downsmapling_layer_with_FPS(nn.Module):
    def __init__(self, out_npoint=256, input_channels= 128, output_channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=1)
        self.out_npoint = out_npoint
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_channels)


    def forward(self, xyz, features):
        fp_index = pointnet2_utils.furthest_point_sample(xyz, self.out_npoint)
        out_feat = pointnet2_utils.gather_operation(features, fp_index)
        out_feat = self.activation(self.bn(self.conv1(out_feat)))
        return out_feat




if __name__ == "__main__":
    sim_data = Variable(torch.rand(2, 1024, 6))
    sim_data = sim_data.cuda(3)
    #sim_cls = Variable(torch.from_numpy(np.arange(32)))
    #sim_cls = sim_cls.cuda()

    seg = RSCNN_MS(num_classes=2, input_channels=3, use_xyz=True)
    seg = seg.cuda()
    out = seg(sim_data)
    print('feature size', out.size())