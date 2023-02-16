import os
import torch
import importlib
import torchvision.transforms as transforms
from PIL import Image
import PIL

"""   patch  match   """

def batch_meshgrid(shape,device):
    batch_size,_,height,width = shape
    x_range = torch.arange(0.0,width,device=device) #return a tensor step = 1  "coordinate range"
    y_range = torch.arange(0.0,height,device=device)
    x_coordinate,y_coordinate = torch.meshgrid(x_range,y_range)  #x_coordinate : in grid[size(x,y)],each row elements are same!  and in y_coordinate: each cols elements are same!
    # so we have two grid each of them is original grid's X_coordinate and Y_coordinate
    x_coordinate = x_coordinate.expand(batch_size,-1,-1).unsqueeze(1)
    y_coordinate = y_coordinate.expand(batch_size,-1,-1).unsqueeze(1)  # -1 means no change in that dimension """ perhaps copy batch_size example """
    return x_coordinate,y_coordinate
    pass

""" offset 1 x 3*16 x 16 x 16 """
def offset_to_inds(offset_x,offset_y):
    shape = offset_x.size()
    device = offset_x.device
    x_coordinate,y_coordinate = batch_meshgrid(shape,device)
    h , w = offset_x.size()[2:]
    """ clamp input(coordinate) to min(0) and max(h/w - 1)"""
    x = torch.clamp(x_coordinate + offset_x, 0, h-1)
    y = torch.clamp(y_coordinate + offset_y, 0, w-1)
    """ now we get every instance for all pixel ,we should find  to which location """
    """ x is really x coordinate in h x w, and so y is """
    return x * offset_x.size()[3] + y
    pass

def inds_to_offset(inds):
    """inds ::  b x number x h x w  """
    shape = inds.size()
    device = inds.device
    x_coordinate,y_coordinate = batch_meshgrid(shape,device) #get (x,y)grids
    batch_size,_,height,width = shape
    x = inds // width  # all elem in inds divid width so that all elem transfer to x_coordinate
    y = inds %  width
    """ x.size() = b , number , h , w"""
    """ x_corrdinate.size() = b , 1 , h , w """ 
    return x - x_coordinate, y - y_coordinate #so there is a broadcast appear in dim = 1
    pass

def feature_normalize(feature_in,eps = 1e-10):
    feature_in_norm = torch.norm(feature_in,2,1,keepdim=True) + eps #calculate norm on C
    feature_in_norm = torch.div(feature_in,feature_in_norm) #div norm to make lenth to be 1
    return feature_in_norm

""" trans tensor to RGB and save """

def tensor_to_RGB(ts,path = ''):
    toPIL = transforms.ToPILImage()
    ts = ts[0,:,:,:]
    if ts.size()[0] == 1:
        ts = ts.squeeze(0)
    ts = ts * 0.5 + 0.5
    plc = toPIL(ts)
    plc.save(path)
    pass

def vgg_preprocess(tensor,vgg_normal_correct = False):
    if vgg_normal_correct:
        tensor = (tensor+1)/2
    
    tensor_bgr = torch.cat((tensor[:,2:3,:,:],tensor[:,1:2,:,:],tensor[:,0:1,:,:]),dim=1)
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392,0.45795686,0.48501961]).type_as(tensor_bgr).view(1,3,1,1)
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst #[0,255]
    pass

def save_network(net,label,epoch):
    save_filename = '%s_net_%s.pth' %(epoch,label)
    save_path = os.path.join('./checkpoints','_coco2_',save_filename)
    torch.save(net.cpu().state_dict(),save_path)
    net.to("cuda:4")
    pass

def load_network(net,label,epoch):
    save_fliename = '%s_net_%s.pth' %(epoch,label)
    save_dir = os.path.join('./checkpoints','_coco2_')
    save_path = os.path.join(save_dir,save_fliename)
    if not os.path.exists(save_path):
        print('not find model :' + save_path + ', do not load model!')
        return net
    weights = torch.load(save_path)
    try:
        net.load_state_dict(weights)
    except KeyError:
        print('key error, not load!')
    except RuntimeError as err:
        print(err)
        net.load_state_dict(weights, strict=False)
        print('loaded with strict = False')
    print('Load from ' + save_path)
    return net
    pass

def weighted_l1_loss(input , target ,weights):
    out = torch.abs(input - target)
    out = out * weights.expand_as(out)
    loss = out.mean()
    return loss

def mse_loss(input , target = 0):
    return torch.mean((input - target) **  2)

def print_latest_losses(epoch, i , num ,errors,t):
    message = '(epoch: %d, iters: %d, finish: %.2f%%, time: %.3f) ' % (epoch, i, (i/num)*100.0, t)
    for k,v in errors.items():
        v = v.mean().float()
        message += '%s: %.3f ' %(k ,v)
    message += '\r'
    print(message,end='')
    log_name = os.path.join('./checkpoints', '_coco2_', 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)
