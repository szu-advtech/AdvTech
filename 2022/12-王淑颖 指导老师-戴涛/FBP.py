import os
import astra
import torch
import numpy as np


net_name = 'unet4recon'
transfer_name = 'unet4recon'
copy_model = False
use_cuda = torch.cuda.is_available()
WaterAtValue = 0.0192
source_root_path = '/mnt/nas/wsy/MayoData'  ## data path
target_root_path = '/mnt/nas/wsy/Unet4reconResult'  ## results path

TrainFolder = {'patients': ['L096', 'L109', 'L143', 'L192', 'L291', 'L310', 'L333', 'L506'],
               'normal_dose': ['full_1mm'], 'low_dose': ['quarter_1mm']}
ValFolder = {'patients': ['L286'], 'normal_dose': ['full_1mm'], 'low_dose': ['quarter_1mm']}
TestFolder = {'patients': ['L067'], 'normal_dose': ['full_1mm'], 'low_dose': ['quarter_1mm']}
results_folder = '{}_result'.format(net_name)

for target_folder in ['Model_save', 'Loss_save', 'Optimizer_save']:
    if not os.path.isdir('{}/{}'.format(target_root_path, target_folder)):
        os.makedirs('{}/{}'.format(target_root_path, target_folder))

batch_num = {'train': 1000, 'val': 100, 'test': 1}
batch_size = {'train': 1, 'val': 1, 'test': 1}
dataloaders_batch_size = {'train': 1, 'val': 1, 'test': 1}
num_workers = {'train': 48, 'val': 10, 'test': 10}
reload_mode = 'val'
is_train = True
reload_model = False
is_lr_scheduler = False
geo_mode = 'fanflat'  # or 'parallel'
angle_range = {'fanflat': 2 * np.pi, 'parallel': np.pi}
is_shuffle = True if is_train else False
num_filters = 4
gpu_id_conv = 0
gpu_id_bp = 0

geo_full = {'nVoxelX': 416, 'sVoxelX': 340.0192, 'dVoxelX': 0.6641,
            'nVoxelY': 416, 'sVoxelY': 340.0192, 'dVoxelY': 0.6641,
            'nDetecU': 640, 'sDetecU': 504.0128, 'dDetecU': 0.6848,
            'views': 640, 'slices': 1,
            'DSD': 600.0, 'DSO': 550.0, 'DOD': 50.0,
            'start_angle': 0.0, 'end_angle': angle_range[geo_mode],
            'mode': geo_mode
            }


class voxel_backprojection(object):
    def __init__(self, geo):
        self.geo = geo
        # create_vol_geom(Y,X,minx,maxx,miny,maxy) A 2D volume geometry of size (Y,X),windowed as minx<=x<=maxx and miny<=y<=maxy
        self.vol_geom = astra.create_vol_geom(geo['nVoxelY'], geo['nVoxelX'],
                                              -1 * geo['sVoxelY'] / 2, geo['sVoxelY'] / 2, -1 * geo['sVoxelX'] / 2,
                                              geo['sVoxelX'] / 2)
        # A fan-beam projection geometry.
        self.proj_geom = astra.create_proj_geom(geo['mode'], geo['dDetecU'], geo['nDetecU'],
                                                np.linspace(geo['start_angle'], geo['end_angle'], geo['views'], False),
                                                geo['DSO'], geo['DOD'])
        # Create a 2D or 3D projector. return The ID of the projector.
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)  # line_fanflat

        self.cfg = astra.astra_dict('FBP_CUDA')
        self.cfg['ProjectorId'] = self.proj_id

    def __call__(self, sinogram):
        sinogram = sinogram.view(self.geo['views'], self.geo['nDetecU'])
        ##
        sino_id = astra.data2d.create('-sino', self.proj_geom)
        astra.data2d.store(sino_id, sinogram.detach().cpu().numpy())  # Fill existing 2D object with data.
        rec_id = astra.data2d.create('-vol', self.vol_geom)

        self.cfg['ReconstructionDataId'] = rec_id
        self.cfg['ProjectionDataId'] = sino_id

        alg_id = astra.algorithm.create(self.cfg)  # FBP #Create algorithm object.return ID
        astra.algorithm.run(alg_id)  # Run an algorithm

        recon = astra.data2d.get(rec_id)  # Get a 2D object.
        astra.data2d.delete(rec_id)  # Delete a 2D object.
        astra.data2d.delete(sino_id)
        astra.algorithm.delete(alg_id)
        ##
        return torch.tensor(recon).view(-1, 1, self.geo['nVoxelX'], self.geo['nVoxelY']).cuda()


class siddon_ray_projection(object):
    def __init__(self, geo):
        self.geo = geo

        self.vol_geom = astra.create_vol_geom(geo['nVoxelY'], geo['nVoxelX'],
                                              -1 * geo['sVoxelY'] / 2, geo['sVoxelY'] / 2, -1 * geo['sVoxelX'] / 2,
                                              geo['sVoxelX'] / 2)
        self.proj_geom = astra.create_proj_geom(geo['mode'], geo['dDetecU'], geo['nDetecU'],
                                                np.linspace(geo['start_angle'], geo['end_angle'], geo['views'], False),
                                                geo['DSO'], geo['DOD'])
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)  # line_fanflat

    def __call__(self, image):
        image = image.view(self.geo['nVoxelX'], self.geo['nVoxelY'])
        sinogram_id, sinogram = astra.create_sino(image.cpu().numpy(),
                                                  self.proj_id)  # Create a forward projection of an image (2D).
        astra.data2d.delete(sinogram_id)

        return torch.tensor(sinogram).view(-1, 1, self.geo['views'], self.geo['nDetecU']).cuda()

