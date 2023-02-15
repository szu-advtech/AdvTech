import os
import open3d as o3d
import numpy as np
import torch
import trimesh

base = './data/ModelNet40/person/train/'
out_path = './data/modelnetply/person/train/'

def to_ply(input_path, output_path, original_type):
    mesh = trimesh.load(input_path, file_type=original_type)  # read file
    mesh.export(output_path, file_type='ply')


def convert():
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for info in os.listdir(base):
        name = info.split('.',1)
        inpath = os.path.join(base,info)
        outpath = out_path+name[0]+'.ply'
    
        # outpath = outpath + '.ply'
        to_ply(inpath,outpath,'off')

if __name__ =='__main__':
    convert()
        