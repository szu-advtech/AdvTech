import os
import open3d as o3d
import numpy as np
import torch
import tqdm

base = './data/modelnetply/person/train/'

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def ply2tensor(file_dir,draw = False):

    pcd_load = o3d.io.read_point_cloud(file_dir)
    if draw:
        o3d.visualization.draw_geometries([pcd_load])
    
    xyz_load = np.asarray(pcd_load.points)
    xyz_tensor = torch.tensor(xyz_load,dtype = torch.float32)
    return xyz_tensor

def obj_2_tensor_or_ply(file_dir,point_num,tensor_or_ply):

    mesh = o3d.io.read_triangle_mesh(file_dir)
    pcd = mesh.sample_points_poisson_disk(point_num)
    if tensor_or_ply == 1:
        xyz_np = np.asarray(pcd.points)
        return torch.tensor(xyz_np,dtype = torch.float32)
    else:
        return pcd

def numpy2ply(xyz,save_dir,draw = False):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(save_dir,pcd)

    if draw:
        pcd_load = o3d.io.read_point_cloud(save_dir)
        o3d.visualization.draw_geometries([pcd_load])

def load_data1(class_num,index1,nums,point_num):
    meshs = []
    tens = []
    for info,bs in zip(os.listdir(base),range(class_num)):
        path = os.path.join(base,info)
        mesh = o3d.io.read_triangle_mesh(path)
        meshs.append(mesh)

    for bs in range(nums):
        pcd = meshs[index1].sample_points_poisson_disk(point_num)
        xyz_np = np.asarray(pcd.points)
        xyz_np = pc_normalize(xyz_np)
        tens.append(xyz_np)

    return torch.tensor(tens,dtype = torch.float32)

def load_data2(class_num,index1,index2,nums,point_num):
    meshs = []
    tens = []
    for info,bs in zip(os.listdir(base),range(class_num)):
        path = os.path.join(base,info)
        mesh = o3d.io.read_triangle_mesh(path)
        meshs.append(mesh)

    for bs in range(nums//2):
        pcd = meshs[index1].sample_points_poisson_disk(point_num)
        xyz_np = np.asarray(pcd.points)
        xyz_np = pc_normalize(xyz_np)
        tens.append(xyz_np)

    for bs in range(nums//2):
        pcd = meshs[index2].sample_points_poisson_disk(point_num)
        xyz_np = np.asarray(pcd.points)
        xyz_np = pc_normalize(xyz_np)
        tens.append(xyz_np) 

    return torch.tensor(tens,dtype = torch.float32)

def load_data_all(point_num):
    meshs = []
    tens = []
    for info in os.listdir(base):
        path = os.path.join(base,info)
        mesh = o3d.io.read_triangle_mesh(path)
        meshs.append(mesh)

    for mesh in meshs:
        pcd = mesh.sample_points_poisson_disk(point_num)
        xyz_np = np.asarray(pcd.points)
        xyz_np = pc_normalize(xyz_np)
        tens.append(xyz_np)

    return torch.tensor(tens,dtype = torch.float32)

def view_data(nums,point_num):
    tens = []
    for info,bs in zip(os.listdir(base),range(nums)):
        path = os.path.join(base,info)
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = o3d.geometry.PointCloud()
        pcd = mesh.sample_points_poisson_disk(point_num)
        tens.append(pcd)
    o3d.visualization.draw_geometries(tens)

if __name__ == '__main__':
    view_data(1,2048)
