import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from utils.util import batch_meshgrid as batch_meshgrid
from utils.util import inds_to_offset as inds_to_offset
from utils.util import offset_to_inds as offset_to_inds

from model.network.convgru import BasicUpdateBlock as BasicUpdateBlock

class Evaluate(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.filter_size = 3
        self.temperature = temperature

    """ now what input is ?? """
    """
    left_features  :  b x c x hw
    right_features :  c x bhw
    offset_x       :  b x 3*windows**2 x h x w
    offset_y       :  b x 3*windows**2 x h x w
    """
    def forward(self,left_features,right_features,offset_x,offset_y):
        device = left_features.get_device()
        batch_size,num,height,width = offset_x.size()
        channel = left_features.size()[1]
        matching_inds = offset_to_inds(offset_x,offset_y) # b x 3*16 x h x w
        matching_inds = matching_inds.view(batch_size,num,height*width).permute(0 , 2 , 1 ).long() # b x hw x 3*16  // offset
        base_batch = torch.arange(batch_size).to(device).long() * (height * width) # size = b , value = 0 , hw , 2*hw ... (b-1)*hw
        base_batch = base_batch.view(-1, 1, 1) # // original coordinate
        """ b x hw x 3*16 + b x 1 x 1 """
        """ for b = 1, that means base_batch = 0 """
        matching_inds_add_base = matching_inds + base_batch
        right_features_view = right_features
        match_cost = []
        #using A[:,idx]
        #for every 3 * 16
        for i in range(matching_inds_add_base.size()[-1]):
            idx = matching_inds_add_base[:, :, i]
            idx = idx.contiguous().view(-1) # vector b x hw expose to bhw each elem in idx is the location index pixel we should search
            right_features_select = right_features_view[:,idx] # we pick the col_vector with idx in size "C"
            """ each col_vector in right_features_select is the feature we should search and compare to the best_match(now) in space pixel """  
            right_features_select = right_features_select.view(channel,batch_size,-1).transpose(0 , 1)
            """ b x c x hw * b x c x hw  ---> elements product """
            """ elements product and then sum in dim = 1, is equal to in dim = 1 process vector dot """
            match_cost_i = torch.sum(left_features * right_features_select , dim=1,keepdim=True)/self.temperature
            """ that's all dot result in  1/(3*16) """
            match_cost.append(match_cost_i)
        """ match_cost.size() = [b x 1 x hw, ....] """
        """ b x 3*16 x hw -> b x hw x 3*16 """
        match_cost = torch.cat(match_cost , dim=1).transpose(1,2)
        """ cat in channel,each elem in list of match_cost is a offset method's cost for match """
        """ normalize vector(hw)'s lenth to 1 """
        match_cost = F.softmax(match_cost,dim=1)
        """ take num//self.filter max elem and its index in dim = 2  '(3*16)' """
        """ now we save top 16 cost in first output ,and top 16 index in second output """
        """ dot result more large, two feature vector more match """
        match_cost_topk,match_cost_topk_Indices = torch.topk(match_cost,num//self.filter_size,dim=-1)
        """ matching_inds b x hw x 3*16 , match_cost_topk_indices b x hw x 16(best 16 match index of offset) ----------offset gather with offset's index """
        """ matching_inds b x hw x 16 , best 16 match offset """
        matching_inds = torch.gather(matching_inds,-1,match_cost_topk_Indices)
        matching_inds = matching_inds.permute(0,2,1).view(batch_size,-1,height,width).float() #resize to (b x 16 x h x w)
        offset_x,offset_y = inds_to_offset(matching_inds)
        corr = match_cost_topk.permute(0,2,1) #resize to b x 16 x hw
        return offset_x,offset_y,corr
        pass

class Propagation(nn.Module):
    def __init__(self):
        super().__init__()
    
    """ offset   b x windows**2 x h x w"""
    def forward(self,offset_x,offset_y,propagation_type = "horizontal"):
        device = offset_x.get_device()
        """ B x c x h x 1"""
        self.horizontal_zeros = torch.zeros((offset_x.size()[0],offset_x.size()[1],offset_x.size()[2],1)).to(device) # y anix set to 1
        """ B x c x 1 x w """
        self.vertical_zeros = torch.zeros((offset_y.size()[0],offset_y.size()[1], 1 , offset_y.size()[3])).to(device) # x anix set to 1
        if propagation_type is "horizontal":
            """  how to cat zeros and offset  """
            """
            in offset_x, each elems in same rows are same
            first , cat zero to each rows and delete the last elem
            for example '1 1 1 1' to '0 1 1 1'
            second , cat '1 1 1 1' to '1 1 1 0'
            last cat in dim = 1  (windows**2)
            so windows*windows method to 3 * windows * windows
            for horizontal , offset_y haddle in same way
            so when offset_x == offset_y == 0, means that pixel only can find itself
            """
            offset_x = torch.cat(
                (
                    torch.cat((self.horizontal_zeros,offset_x[:,:,:,:-1]) , dim=3),
                    offset_x,
                    torch.cat((offset_x[:,:,:,1:],self.horizontal_zeros),dim=3)
                ),
                dim=1
            )
            offset_y = torch.cat((
                torch.cat((self.horizontal_zeros,offset_y[:,:,:,:-1]),dim=3),
                offset_y,
                torch.cat((offset_y[:,:,:,1:],self.horizontal_zeros), dim=3)
            ),dim=1)
            """  how to use this kind of offset  ????  """
        else:
            offset_x = torch.cat((
                torch.cat((self.vertical_zeros,offset_x[:,:,:-1,:]),dim=2),
                offset_x,
                torch.cat((offset_x[: , : , 1: , :],self.vertical_zeros),dim=2)
            ),dim=1)
            offset_y = torch.cat((
                torch.cat((self.vertical_zeros,offset_y[:,:,:-1,:]),dim=2),
                offset_y,
                torch.cat((offset_y[: , : , 1: , :],self.vertical_zeros),dim=2)
            ),dim=1)
        return offset_x, offset_y
        pass

class PatchMatchOnce(nn.Module):
    def __init__(self):
        super().__init__()
        self.propagation = Propagation()
        self.evaluate = Evaluate(0.01)
    """ what is input ???  """
    """
    left_features : f1 , content_feature  b x c x h*w
    right_feature:  f2_view . refer_feature_view  C x b*h*w
    offset_x : every_pixel_position 's offset in x anix  B x windows**2 x h x w
    """
    def forward(self,left_features,right_features,offset_x,offset_y):
        prob = random.random()
        if prob < 0.5:
            offset_x,offset_y = self.propagation(offset_x,offset_y,"horizontal")  #search nieghber 16 pixel as new offset and cat to 16*3
            offset_x,offset_y,_ = self.evaluate(left_features,right_features,offset_x,offset_y) #from 16*3 kinds of offset calculate best 16 match offset
            offset_x,offset_y = self.propagation(offset_x,offset_y,"vertical")
            offset_x,offset_y,corr = self.evaluate(left_features,right_features,offset_x,offset_y)
        else:
            offset_x,offset_y = self.propagation(offset_x,offset_y,"vertical")  
            offset_x,offset_y,_ = self.evaluate(left_features,right_features,offset_x,offset_y) 
            offset_x,offset_y = self.propagation(offset_x,offset_y,"horizontal")
            offset_x,offset_y,corr = self.evaluate(left_features,right_features,offset_x,offset_y)
            pass
        """
        offset_x,offset_y -> ' b x window**2 x h x w '
        corr -> ' b x window**2 x h x w '
        """
        return offset_x,offset_y,corr
        pass

class PatchMatchGRU(nn.Module):
    def __init__(self,temperature,iters,input_dim):
        super().__init__()
        self.patch_match_once_step = PatchMatchOnce()
        self.temperature = temperature
        self.iters = iters
        self.input_dim = input_dim
        hidden_dim = 32
        norm = nn.InstanceNorm2d(hidden_dim,affine=False)
        relu = nn.ReLU(inplace=True)
        """
        concat left and right feature
        """
        self.initial_layer = nn.Sequential(
            nn.Conv2d(input_dim*2,hidden_dim,kernel_size=3,padding=1,stride=1),
            norm,
            relu,
            nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,padding=1),
            norm,
            relu
        )
        self.refine_net = BasicUpdateBlock()

    def forward(self,left_features,right_features,right_input,initial_offset_x,initial_offset_y):
        device = left_features.get_device()
        batch_size,channel,height,width = left_features.size()
        num = initial_offset_x.size()[1] #windows x windows
        initial_input = torch.cat((left_features,right_features),dim=1) # we combine the feature rather the offset
        hidden = self.initial_layer(initial_input) # use initial_layer to initial two concat feature_map
        # and then  how to use that hidden  ??
        left_features = left_features.view(batch_size,-1,height*width) #zhan kai
        right_features = right_features.view(batch_size,-1,height*width)
        right_features_view = right_features.transpose(0,1).contiguous().view(channel,-1) #let channel to the first position  and b,h*w -> b*h*w
        with torch.no_grad():
            offset_x,offset_y = initial_offset_x,initial_offset_y
        """ start iter  """
        for it in range(self.iters):
            with torch.no_grad():
                offset_x,offset_y,corr = self.patch_match_once_step(left_features,right_features_view,offset_x,offset_y)
            """ GRU refinement """
            flow = torch.cat((offset_x,offset_y),dim=1)
            corr = corr.view(batch_size,-1,height,width)
            hidden,delta_offset_x,delta_offset_y = self.refine_net(hidden,corr,flow)
            offset_x = offset_x + delta_offset_x
            offset_y = offset_y + delta_offset_y
            with torch.no_grad():
                matching_inds = offset_to_inds(offset_x,offset_y)
                matching_inds = matching_inds.view(batch_size,num,height*width).permute(0,2,1).long()
                base_batch = torch.arange(batch_size).to(device).long() * (height*width)
                base_batch = base_batch.view(-1,1,1)
                matching_inds_plus_base = matching_inds + base_batch
        match_cost = []
        #using A[:,idx]
        for i in range(matching_inds_plus_base.size()[-1]): # 0 ~ num-1  
            idx = matching_inds_plus_base[:,:,1]
            idx = idx.contiguous().view(-1)
            right_features_select = right_features_view[:,idx]
            right_features_select = right_features_select.view(channel,batch_size,-1).transpose(0,1)
            match_cost_i = torch.sum(left_features*right_features_select,dim=1,keepdim=True)/self.temperature
            match_cost.append(match_cost_i)
        match_cost = torch.cat(match_cost,dim=1).transpose(1,2)
        match_cost = F.softmax(match_cost,dim=-1)  # 16 different kinds of offset and softmax in hw !!!!!!!!!! trick!!!
        right_input_view = right_input.transpose(0,1).contiguous().view(right_input.size()[1],-1)  # c x bhw
        warp = torch.zeros_like(right_input)
        #using A[:,ids]
        for i in range(match_cost.size()[-1]):
            idx = matching_inds_plus_base[:,:,i]
            idx = idx.contiguous().view(-1)
            right_input_select = right_input_view[:,idx]
            right_input_select = right_input_select.view(right_input.size()[1],batch_size,-1).transpose(0,1)
            warp = warp + right_input_select * match_cost[:,:,i].unsqueeze(dim = 1)
        """
        matching_inds : b x hw x 16
        right_input   : b x 3  x hw
        warp          : b x 3  x hw
        match_cost_i  : b x c  x hw
        right_input_select: c x hw -> b x 3 x hw 
        """
        """ finally 16 kinds of warp result * weight """
        return matching_inds,warp
        pass
    pass