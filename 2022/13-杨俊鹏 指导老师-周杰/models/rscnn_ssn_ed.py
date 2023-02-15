import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules import PointnetSAModule, PointnetFPModule, PointnetSAModuleMSG
import numpy as np

class RSCNN_MSN(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Encoder-Decoder network that uses feature propogation layers

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

    def __init__(self, num_classes, input_channels=0, relation_prior=1, use_xyz=True, training=True):
        super().__init__()

        self.training = training

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(     # 0
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.23],
                nsamples=[48],
                mlps=[[c_in, 128]],
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_0 = 128

        c_in = c_out_0
        self.SA_modules.append(    # 1
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.32],
                nsamples=[64],
                mlps=[[c_in, 256]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_1 = 256

        c_in = c_out_1
        self.SA_modules.append(    # 2
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.32],
                nsamples=[64],
                mlps=[[c_in, 512]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_2 = 512

        #c_in = c_out_2
        #self.SA_modules.append(    # 3
        #    PointnetSAModuleMSG(
        #        npoint=16,
        #        radii=[0.4, 0.6, 0.8],
        #        nsamples=[16, 24, 32],
        #        mlps=[[c_in, 512], [c_in, 512], [c_in, 512]],
        #        use_xyz=use_xyz,
        #        relation_prior=relation_prior
        #    )
        #)
        #c_out_3 = 512*3
        
        self.SA_modules.append(   # 4   global pooling
            PointnetSAModule(
                nsample = 256,
                mlp=[c_out_2, 1024], 
                use_xyz=use_xyz
            )
        )
        global_out = 1024
        
        #self.SA_modules.append(   # 5   global pooling
        #    PointnetSAModule(
        #        nsample = 256,
        #        mlp=[c_out_2, 128], use_xyz=use_xyz
        #    )
        #)
        #global_out2 = 256

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointnetFPModule(mlp=[512 + input_channels, 256])
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[1024 + c_out_0, 512])
        )
        #self.FP_modules.append(PointnetFPModule(mlp=[1024 + c_out_1, 512]))
        self.FP_modules.append(
            PointnetFPModule(mlp=[c_out_1 + c_out_2, 1024])
        )

        self.SA_encoder = nn.ModuleList()
        self.SA_encoder.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.32],
                nsamples=[64],
                mlps=[[256, 256]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        self.SA_encoder.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.48],
                nsamples=[64],
                mlps=[[256, 512]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        self.SA_encoder.append(
            PointnetSAModule(
            nsample=256,
            mlp=[512, 1024],
            use_xyz=use_xyz
            )
        )

        self.mid_FC_layer = nn.Sequential(
            pt_utils.FC(1024, 512, activation=nn.ReLU(), bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(512, 512, activation=nn.ReLU(), bn=True),
            nn.Dropout(p=0.5),
        )
        #self.midlinear = pt_utils.FC(512. num_classes, activation=None)

        self.FC_layer = nn.Sequential(
            #pt_utils.Conv1d(128+global_out+global_out2+16, 128, bn=True), nn.Dropout(),
            #pt_utils.Conv1d(128, num_classes, activation=None)
            pt_utils.FC(1024, 512, activation=nn.ReLU(), bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(512, 512, activation=nn.ReLU(), bn=True),
            nn.Dropout(p=0.5),
        )
        self.finallinear = pt_utils.FC(512, num_classes, activation=None)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
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
        
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            if i < 4:
                li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                #if li_xyz is not None:
                #    random_index = np.arange(li_xyz.size()[1])
                #    np.random.shuffle(random_index)
                #    li_xyz = li_xyz[:, random_index, :]
                #    li_features = li_features[:, :, random_index]
                l_xyz.append(li_xyz)
                l_features.append(li_features)
        
        #_, global_out_feat_mid = self.SA_modules[3](l_xyz[2], l_features[2])
        global_out_feat_mid = l_features[-1]
        if self.training:
            global_out_feat_mid = self.mid_FC_layer(global_out_feat_mid.squeeze(-1))

        
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1 - 1] = self.FP_modules[i](
                l_xyz[i - 1 - 1], l_xyz[i - 1], l_features[i - 1 - 1], l_features[i - 1]
            )

        out_xyz, out_features = l_xyz[0], l_features[0]
        for module in self.SA_encoder:
            out_xyz, out_features = module(out_xyz, out_features)
        mediate_features = self.FC_layer(out_features.squeeze(-1))
        if self.training:
            return mediate_features, self.finalLinear(mediate_features), global_out_feat_mid
        else:
            return mediate_features

if __name__ == '__main__':
    x = torch.randn(2, 1024, 3).cuda()
    rscnn_encoder_decoder = RSCNN_MSN(8,training=False).cuda()
    rscnn_encoder_decoder.eval()
    y = rscnn_encoder_decoder(x)
    print(y.shape)