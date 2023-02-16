from dis import dis
from math import dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from torchsummary import summary
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.pointnet import PointNetEncoder
from models.graph import GraphTripleConv, GraphTripleConvNet
from models.pointnetxt import PointNextEncoder
from config import CONF


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
	layers = []
	for i in range(len(dim_list) - 1):
		dim_in, dim_out = dim_list[i], dim_list[i + 1]
		layers.append(nn.Linear(dim_in, dim_out))
		final_layer = (i == len(dim_list) - 2)
		if not final_layer or final_nonlinearity:
			if batch_norm == 'batch':
				layers.append(nn.BatchNorm1d(dim_out))
			if activation == 'relu':
				layers.append(nn.ReLU())
			elif activation == 'leakyrelu':
				layers.append(nn.LeakyReLU())
		if dropout > 0:
			layers.append(nn.Dropout(p=dropout))
	return nn.Sequential(*layers)


def load_pretained_cls_model(model, local_rank, name):
    # load pretrained pointnet_cls model [ relPointNet ver. ]
    if name == "object":
        pretrained_dict = torch.load(os.path.join(CONF.PATH.BASE, './SSG_pointobject_ckpt_best.pth'), map_location=lambda storage, loc: storage.cuda(local_rank))["model"]
    else:
        pretrained_dict = torch.load(os.path.join(CONF.PATH.BASE, './SSG_pointrelation_ckpt_latest.pth'), map_location=lambda storage, loc: storage.cuda(local_rank))["model"]
    net_state_dict = model.state_dict()
    pretrained_dict_ = {k[8:]: v for k, v in pretrained_dict.items() if 'encoder' in k and v.size() == net_state_dict[k[8:]].size()}
    a = []
    for i in pretrained_dict_.keys():
        if i not in net_state_dict.keys():
            a.append(i)
    b = []
    for j in net_state_dict.keys():
        if j not in pretrained_dict_.keys():
            b.append(j)
    net_state_dict.update(pretrained_dict_)
    model.load_state_dict(net_state_dict)



class SGPN(nn.Module):
    def __init__(self, use_pretrained_cls, gconv_dim=128, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5, mlp_normalization='none',
                 obj_cat_num=160, pred_cat_num=27):
        super().__init__()

        # ObjPointNet and RelPointNet
        local_rank = torch.distributed.get_rank()
        
        self.objPointNet = PointNextEncoder(in_channels=3) # (x,y,z)
        
        # summary(self.objPointNet, input_size=[(1000, 3)], batch_size=9, device="cpu")
        
        if use_pretrained_cls:
            load_pretained_cls_model(self.objPointNet.cuda(local_rank), local_rank, "object")

        self.relPointNet = PointNextEncoder(in_channels=4) # (x,y,z,M) M-> class-agnostic instance segmentation
        
        # summary(self.relPointNet, input_size=[(3000, 3), (3000, 4)], batch_size=72, device="cpu")
        
        # if use_pretrained_cls:
            # load_pretained_cls_model(self.relPointNet.cuda(local_rank), local_rank, "relation")

        # GCN module
        if gconv_num_layers == 0:
            self.gconv = nn.Linear(512, gconv_dim) # final feature of the pointNet2
        elif gconv_num_layers > 0:
            gconv_kwargs = {
                'input_dim': 512,
                'output_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
                'input_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers - 1,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        # MLP for classification
        obj_classifier_layer = [gconv_dim, 256, obj_cat_num]
        self.obj_classifier = build_mlp(obj_classifier_layer, batch_norm=mlp_normalization)

        rel_classifier_layer = [gconv_dim, 256, pred_cat_num]
        self.rel_classifier = build_mlp(rel_classifier_layer, batch_norm=mlp_normalization)

    def forward(self, data_dict):
        objects_id = data_dict["objects_id"]
        objects_pc = data_dict["objects_pc"]
        objects_count = data_dict["aligned_obj_num"]   # namely 9
        predicate_pc_flag = data_dict["predicate_pc_flag"]  # [num_rel, predicate_sample_dim, 4] 4表示点+（sub/obj/background）
        predicate_count = data_dict["aligned_rel_num"] # namely 72
        edges = data_dict["edges"]
        batch_size = objects_id.size(0)

        # point cloud pass objPointNet
        # objects_pc = objects_pc.permute(0, 2, 1)  # [num_object, 3, points_num]
        obj_vecs = self.objPointNet(objects_pc)  # obj_vecs[num_object, H] tf1[object_num, 64, 64]表示转换矩阵（STNkd）

        # point cloud pass relPointNet
        # predicate_pc_flag = predicate_pc_flag.permute(0, 2, 1)  # [num_predicate, 4, predicate_sample_dim]
        pred_vecs = self.relPointNet(predicate_pc_flag[:, :, :3].contiguous(), predicate_pc_flag.transpose(1, 2).contiguous())

        # obj_vecs and rel_vecs pass GCN module
        obj_vecs_list = []
        pred_vecs_list = []
        object_num = int(objects_count.item())
        predicate_num = int(predicate_count.item())
        for i in range(batch_size):
            if isinstance(self.gconv, nn.Linear):
                o_vecs = self.gconv(obj_vecs[object_num*i: object_num*(i+1)])
            else:
                o_vecs, p_vecs = self.gconv(obj_vecs[object_num*i: object_num*(i+1)], pred_vecs[predicate_num*i: predicate_num*(i+1)], edges[i])
            if self.gconv_net is not None:
                o_vecs, p_vecs = self.gconv_net(o_vecs, p_vecs, edges[i])  # 多层gconv

            obj_vecs_list.append(o_vecs)
            pred_vecs_list.append(p_vecs)

        obj_pred_list = []
        rel_pred_list = []  # 在batch中的每一个点云上进行操作
        for o_vec in obj_vecs_list:
            obj_pred = self.obj_classifier(o_vec)
            obj_pred_list.append(obj_pred)
        for p_vec in pred_vecs_list:
            rel_pred = self.rel_classifier(p_vec)
            rel_pred_list.append(rel_pred)
        # ATTENTION: as batch_size > 1, the value that corresponds to the "predict" key is a list
        data_dict["objects_predict"] = obj_pred_list
        data_dict["predicate_predict"] = rel_pred_list

        return data_dict
