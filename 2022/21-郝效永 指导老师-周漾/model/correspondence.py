import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.util as util
from model.network.BaseModel import BaseNetwork
from model.network.U_net import U_Net
from model.network.patchmatch import PatchMatchGRU
from utils.util import batch_meshgrid as batch_meshgrid
from utils.util import inds_to_offset as inds_to_offset


#feature--channels alians to 64,match_kernel = 1,PONO_C = True(use C normalization in corr module)
def match_kernel_and_pono_c(feature, match_kernel, PONO_C, eps=1e-10):
    b, c, h, w = feature.size()  # 1 64 16 16
    if match_kernel == 1:
        feature = feature.view(b, c, -1) # 1  64  h*w
    else:
        feature = F.unfold(feature, kernel_size=match_kernel, padding=int(match_kernel//2))
    dim_mean = 1 if PONO_C else -1
    feature = feature - feature.mean(dim=dim_mean, keepdim=True)  #calculate means on C
    feature_norm = torch.norm(feature, 2, 1, keepdim=True) + eps
    feature = torch.div(feature, feature_norm) # lenth to 1
    return feature.view(b, -1, h, w)

class Domain_alin(BaseNetwork):
    def __init__(self):
        super().__init__()
        self.adptive_model_seg = U_Net(input_channels=20)
        self.adptive_model_img = U_Net(input_channels=23)

        feature_channel = 64
        "128 * 128"
        self.phi_0 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.phi_1 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.phi_2 = nn.Conv2d(in_channels=feature_channel*2, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.phi_3 = nn.Conv2d(in_channels=feature_channel, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.theta_0 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.theta_1 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.theta_2 = nn.Conv2d(in_channels=feature_channel*2, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.theta_3 = nn.Conv2d(in_channels=feature_channel, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        #self.print_network()
        #self.init_weights()
    
        self.patch_match = PatchMatchGRU(temperature=0.01,iters=5,input_dim=64)
        pass
    """128  *  128"""
    def multi_scale_patch_match(self,f1,f2,ref,hierarchical_scale,pre =None,real_img=None):
        if hierarchical_scale == 0:
            y_circle = None
            scale = 32
            batch_size,channels,feature_height,feature_width = f1.size()
            # ref is 128 * 128
            ref = F.avg_pool2d(ref, 8, stride=8) 
            ref = ref.view(batch_size,3,scale*scale)
            f1 = f1.view(batch_size,channels,scale*scale)
            f2 = f2.view(batch_size,channels,scale*scale)
            matmul_result = torch.matmul(f1.permute(0,2,1),f2)/0.01
            mat = F.softmax(matmul_result,dim=-1)
            y = torch.matmul(mat,ref.permute(0,2,1))

            mat_circle = F.softmax(matmul_result.transpose(1,2),dim=-1)
            y_circle = torch.matmul(mat_circle,y)
            y_circle = y_circle.permute(0,2,1).view(batch_size,3,scale,scale)
            y = y.permute(0,2,1).view(batch_size,3,scale,scale)
            return mat,y,y_circle

        if hierarchical_scale == 1:
            scale = 64
            with torch.no_grad():
                batch_size,channels,feature_height,feature_width = f1.size()
                topk_num = 1
                search_window = 4
                centering = 1
                dilation = 2
                total_candidate_num = topk_num * (search_window ** 2)
                topk_inds = torch.topk(pre,topk_num,dim=-1)[-1]
                inds = topk_inds.permute(0,2,1).view(batch_size,topk_num,(scale//2),(scale//2)).float()
                # offset_*.size()  =  B x 1 x 16 x 16
                offset_x,offset_y = inds_to_offset(inds)
                # dx.size() = search_window  -> search_window x 1  ->  search_window x search_window  -> search_window**2
                # 0 1 2 3  -> 
                """
                0 0 0 0    -1 -1 -1 -1
                1 1 1 1 ->  0  0  0  0
                2 2 2 2     1  1  1  1
                3 3 3 3     2  2  2  2
                why set centering here???
                """
                dx = torch.arange(search_window,dtype=inds.dtype,device=topk_inds.device).unsqueeze_(dim=1).expand(-1,search_window).contiguous().view(-1) - centering 
                dy = torch.arange(search_window,dtype=inds.dtype,device=topk_inds.device).unsqueeze_(dim=0).expand(search_window,-1).contiguous().view(-1) - centering 
                """ 
                dx.size() = 1 x 16 x 1 x 1
                -2 -2 -2 -2 0 0 0 0 2 2 2 2 4 4 4 4
                why set dilation here???
                """
                dx = dx.view(1 , search_window**2 , 1 ,1) * dilation
                dy = dy.view(1 , search_window**2 , 1 ,1) * dilation
                """
                offset_x.size() = B x topk_Num x 16 x 16
                dx.size() = 1 x windows**2 x 1 x 1
                ......
                offset_x*2 + dx .size() =  B x windows**2 x 16 x 16
                so each windows elem is add to offset in h,w(16 x 16)
                this operation create 16 different search method for every position
                scale_factor = 2 , an upsampling on the method
                the tensor (offset_x/y_up) has 1 batch , 16 channels, 16x16 space,every channels provide a different method work on space
                the elem in h x w,can work on pixel, but !!! ````` patch match will search their top,left nighber's offset tensor
                """
                offset_x_up = F.interpolate((2 * offset_x + dx) , scale_factor=2)
                offset_y_up = F.interpolate((2 * offset_y + dy) , scale_factor=2)
            ref = F.avg_pool2d(ref,4,stride = 4)
            ref = ref.view(batch_size,3,scale*scale)
            mat,y = self.patch_match(f1,f2,ref,offset_x_up,offset_y_up)
            y = y.view(batch_size,3,scale,scale)
            return mat,y
        if hierarchical_scale == 2:
            scale = 128
            with torch.no_grad():
                batch_size,channels,feature_height,feature_width = f1.size()
                topk_num = 1
                search_window = 4
                centering = 1
                dilation = 2
                total_candidate_num = topk_num * (search_window**2)
                """ before that, we give top 16 best match method in m(pre),now we select the best match to finish next work """
                topk_inds = pre[:,:,:topk_num]
                inds = topk_inds.permute(0,2,1).view(batch_size,topk_num,(scale//2),(scale//2)).float()
                offset_x,offset_y = inds_to_offset(inds)
                dx = torch.arange(search_window, dtype=topk_inds.dtype, device=topk_inds.device).unsqueeze_(dim=1).expand(-1, search_window).contiguous().view(-1) - centering
                dy = torch.arange(search_window, dtype=topk_inds.dtype, device=topk_inds.device).unsqueeze_(dim=0).expand(search_window, -1).contiguous().view(-1) - centering
                dx = dx.view(1,search_window**2,1,1)*dilation
                dy = dy.view(1,search_window**2,1,1)*dilation
                offset_x_up = F.interpolate((2*offset_x + dx) , scale_factor=2)
                offset_y_up = F.interpolate((2*offset_y + dy) , scale_factor=2)
            ref = F.avg_pool2d(ref , 2,stride = 2)
            ref = ref.view(batch_size,3,scale*scale)
            mat,y = self.patch_match(f1 , f2 ,ref ,offset_x_up ,offset_y_up)
            y = y.view(batch_size,3,scale,scale)
            return mat,y
        if hierarchical_scale == 3:
            scale = 256
            with torch.no_grad():
                batch_size,channels,feature_height,feature_width = f1.size()
                topk_num = 1
                search_window = 4
                centering = 1
                dilation = 2
                total_candidate_num = topk_num * (search_window**2)
                topk_inds = pre[:,:,:topk_num]
                inds = topk_inds.permute(0,2,1).view(batch_size,topk_num,(scale//2),(scale//2)).float()
                offset_x,offset_y = inds_to_offset(inds)
                dx = torch.arange(search_window,dtype=topk_inds.dtype,device=topk_inds.device).unsqueeze_(dim=1).expand(-1,search_window).contiguous().view(-1) - centering
                dy = torch.arange(search_window, dtype=topk_inds.dtype, device=topk_inds.device).unsqueeze_(dim=0).expand(search_window, -1).contiguous().view(-1) - centering
                dx = dx.view(1,search_window**2,1,1)*dilation
                dy = dy.view(1,search_window**2,1,1)*dilation
                offset_x_up = F.interpolate((2*offset_x + dx),scale_factor=2)
                offset_y_up = F.interpolate((2*offset_y + dy),scale_factor=2)
            ref = ref.view(batch_size,3,scale*scale)
            mat,y = self.patch_match(f1,f2,ref,offset_x_up,offset_y_up)
            y = y.view(batch_size,3,scale,scale)
            return mat,y
        pass

    def multi_scale_patch_match_no_patchmatch(self,f1,f2,ref,hierarchical_scale,pre =None,real_img=None):
        if hierarchical_scale == 0:
            y_circle = None
            scale = 16
            batch_size,channels,feature_height,feature_width = f1.size()
            # ref is 128 * 128
            ref = F.avg_pool2d(ref, 8, stride=8) 
            ref = ref.view(batch_size,3,scale*scale)
            f1 = f1.view(batch_size,channels,scale*scale)
            f2 = f2.view(batch_size,channels,scale*scale)
            matmul_result = torch.matmul(f1.permute(0,2,1),f2)/0.01
            mat = F.softmax(matmul_result,dim=-1)
            y = torch.matmul(mat,ref.permute(0,2,1))

            mat_circle = F.softmax(matmul_result.transpose(1,2),dim=-1)
            y_circle = torch.matmul(mat_circle,y)
            y_circle = y_circle.permute(0,2,1).view(batch_size,3,scale,scale)
            y = y.permute(0,2,1).view(batch_size,3,scale,scale)
            return y,y_circle

        if hierarchical_scale == 1:
            scale = 32
            batch_size,channels,feature_height,feature_width = f1.size()
            # ref is 256 * 256
            ref = F.avg_pool2d(ref, 4, stride=4) 
            ref = ref.view(batch_size,3,scale*scale)
            f1 = f1.view(batch_size,channels,scale*scale)
            f2 = f2.view(batch_size,channels,scale*scale)
            matmul_result = torch.matmul(f1.permute(0,2,1),f2)/0.01
            mat = F.softmax(matmul_result,dim=-1)
            y = torch.matmul(mat,ref.permute(0,2,1))
            y = y.permute(0,2,1).view(batch_size,3,scale,scale)
            return y
        if hierarchical_scale == 2:
            scale = 64
            batch_size,channels,feature_height,feature_width = f1.size()
            # ref is 256 * 256
            ref = F.avg_pool2d(ref, 2, stride=2) 
            ref = ref.view(batch_size,3,scale*scale)
            f1 = f1.view(batch_size,channels,scale*scale)
            f2 = f2.view(batch_size,channels,scale*scale)
            matmul_result = torch.matmul(f1.permute(0,2,1),f2)/0.01
            mat = F.softmax(matmul_result,dim=-1)
            y = torch.matmul(mat,ref.permute(0,2,1))
            y = y.permute(0,2,1).view(batch_size,3,scale,scale)
            return y
        if hierarchical_scale == 3:
            scale = 128
            batch_size,channels,feature_height,feature_width = f1.size()
            # ref is 256 * 256
            
            ref = ref.view(batch_size,3,scale*scale)
            f1 = f1.view(batch_size,channels,scale*scale)
            f2 = f2.view(batch_size,channels,scale*scale)
            matmul_result = torch.matmul(f1.permute(0,2,1),f2)/0.01
            mat = F.softmax(matmul_result,dim=-1)
            y = torch.matmul(mat,ref.permute(0,2,1))
            y = y.permute(0,2,1).view(batch_size,3,scale,scale)
            return y
        pass

    def forward(self,ref_img,real_img,ref_label,real_label):
        corr_out = {}
        # set tag
        seg_imput = real_label
        #seg_imput = ref_img

        # set tag
        adptive_seg_feature = self.adptive_model_seg(real_label,real_label)
        #adptive_seg_feature = self.adptive_model_img(seg_imput,seg_imput)

        ref_imput = torch.cat((ref_img,ref_label),dim=1)
        adptive_img_feature = self.adptive_model_img(ref_imput,ref_label)
        for i in range(len(adptive_img_feature)):
            adptive_img_feature[i] = util.feature_normalize(adptive_img_feature[i])
            adptive_seg_feature[i] = util.feature_normalize(adptive_seg_feature[i])

        
        # set tag
        real_input = real_img
        #real_input = ref_img
        rel_input = torch.cat((real_input,real_label),dim=1)
        adptive_img_feature_pair = self.adptive_model_img(rel_input,real_label)
        loss_novgg_featpair = 0.0
        weights = [1.0,1.0,1.0,1.0]
        for i in range(len(adptive_img_feature_pair)):
            adptive_img_feature_pair[i] = util.feature_normalize(adptive_img_feature_pair[i])
            loss_novgg_featpair += F.l1_loss(adptive_seg_feature[i],adptive_img_feature_pair[i])*weights[i]
        #correspondence loss---------
        corr_out['loss_novgg_featpair'] = loss_novgg_featpair * 10.0

        cont_features = adptive_seg_feature
        ref_features = adptive_img_feature
        theta = []
        phi = []
        #four layers to PatchMatch
        theta.append(match_kernel_and_pono_c(self.theta_0(cont_features[0]),1,True))
        theta.append(match_kernel_and_pono_c(self.theta_1(cont_features[1]),1,True))
        theta.append(match_kernel_and_pono_c(self.theta_2(cont_features[2]),1,True))
        theta.append(match_kernel_and_pono_c(self.theta_3(cont_features[3]),1,True))
        phi.append(match_kernel_and_pono_c(self.phi_0(ref_features[0]),1,True))
        phi.append(match_kernel_and_pono_c(self.phi_1(ref_features[1]),1,True))
        phi.append(match_kernel_and_pono_c(self.phi_2(ref_features[2]),1,True))
        phi.append(match_kernel_and_pono_c(self.phi_3(ref_features[3]),1,True))
        
        ref = ref_img
        ys = []
        ms = []
        m = None
        for i in range(len(theta)):
            if i == 0:
                # m is matrix that feature_cont correspondence to feature_ref(softmax)
                # y is target image that ref warp to content_feature
                # y_cycle is ref_ref_image that warp back
                y,y_cycle = self.multi_scale_patch_match_no_patchmatch(theta[i],phi[i],ref,i,pre=m)
                if y_cycle is not None:
                    corr_out["warp_cycle"] = y_cycle
            else:
                y = self.multi_scale_patch_match_no_patchmatch(theta[i],phi[i],ref,i,pre=None)
            ys.append(y)
            # ms.append(m)
        corr_out["warp_out"] = ys
        corr_out['adaptive_feature_seg'] = theta
        corr_out['adaptive_feature_img'] = phi
        corr_out['index_check'] = ms

        return corr_out
        pass
    pass