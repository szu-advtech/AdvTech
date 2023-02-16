from config import opts
import os
import torch
import model
import numpy as np
import open3d as o3d
import sys
sys.path.append(".")
from Common.network_utils import requires_grad
# from Common.pointoperation import normalize_simple_point_cloud

def visualize_sevel(model,x,z,numbers):
    zs = z
    for i in range(numbers):
        zi = model.noise_generator()
        zs = torch.cat((zs,zi),dim=0)
    xs = x.repeat([numbers+1,1,1])
    pcd_s = model.G(xs,zs).detach()
    # pcds = pcd_s.detach()
    visual_pc(pcd_s)
    return pcd_s,zs


def visual_pc(pc_tensor,):
    # pc_tensor = pc_tensor.detach()
    xyz = pc_tensor.transpose(1,2).cpu().numpy()
    b,n,_ = xyz.shape
    one = np.ones([n,1],float)
    zero = np.zeros([n,2],float)
    add = np.concatenate((one,zero),axis = 1)
    pcds = []
    for i in range(b):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz[i]+i*2*add)
        pcds.append(pc)

    o3d.visualization.draw_geometries(pcds)

def show_interpolate(model,x,z,index1,index2,nums = 10):
    alpha = 1/nums
    z_inter = z[index1:index1+1]
    for i in range(nums):
        alpha_inter = i*alpha
        if i!= 0:
            z_i = z[index1:index1+1]*(1-alpha_inter)+z[index2:index2+1]*alpha_inter
            # pcds = model.G(x.repeat([2,1,1]),z_i.repeat([2,1,1])).detach()
            # visual_pc(pcds)

            z_inter = torch.cat((z_inter,z_i),dim = 0)
    x = x.repeat([nums,1,1])
    pcds = model.G(x,z_inter).detach()
    visual_pc(pcds)
    
if __name__ == '__main__':
    opts.pretrain_model_G = "10000_monitor_G.pth"
    # path_save = os.path.join(opts.log_dir,'chair')
    opts.log_dir = "log/20221208-1446"

    model = model.Model(opts)
    model.build_model()
    
    
    path = os.path.join(model.opts.log_dir,model.opts.pretrain_model_G)

    checkpoint = torch.load(path)
    model.G.load_state_dict(checkpoint['G_model'])

    z1 = model.noise_generator()
    x = model.sphere_generator()
    requires_grad(model.G,False)

    pcds,zs = visualize_sevel(model,x[:1],z1,14)
    