import os

import numpy
import numpy as np
import warnings
import pickle

from tqdm import tqdm
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ftModelNetDataLoader(Dataset):
    def __init__(self, root,  split='train', process_data=False):
        self.root = root
        self.npoints = 10000
        self.process_data = process_data
        self.uniform = True
        self.use_normals = False


        self.catfile = os.path.join(self.root, 'new_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))


        assert (split == 'train' or split == 'test')
        shape_names = [line.rstrip() for line in open(os.path.join(self.root, 'new_shape_names.txt'))]
        #self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i]) + '_0001.txt') for i in range(len(shape_names))]
        self.datapath = [
            ('cylinder',self.root+'master_chef_can_0001.txt'),('cylinder',self.root+'tomato_soup_can_0001.txt'),('cylinder',self.root+'large_marker_0001.txt'),
            ('cylinder',self.root+'pitcher_base_0001.txt'),('cylinder',self.root+'mug_0001.txt'),('cylinder',self.root+'mustard_bottle_0001.txt'),('cylinder',self.root+'bleach_cleanser_0001.txt'),
            ('box', self.root+'cracker_box_0001.txt'),('box', self.root+'gelatin_box_0001.txt'),('box', self.root+'pudding_box_0001.txt'),
            ('box', self.root+'sugar_box_0001.txt'),('box', self.root+'potted_meat_can_0001.txt'),('box', self.root+'wood_block_0001.txt'),('box', self.root+'foam_brick_0001.txt'),
            ('flat_cylinder',self.root+'tuna_fish_can_0001.txt'),('bowl',self.root+'bowl_0001.txt'),('banana',self.root+'banana_0001.txt'),
            ('scissors',self.root+'scissors_0001.txt'),('extra_large_clamp',self.root+'extra_large_clamp_0001.txt'),('power_drill',self.root+'power_drill_0001.txt')
        ]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float64)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ftModelNetDataLoader('../data/modelnet40_new_test/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=24, shuffle=False)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
