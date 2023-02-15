import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays
from scipy.io import loadmat


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # [debug]
        # print("trans:", transform)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            self.masks = []
            # self.exps = [] if 'exp_params' in transform['frames'][0] else None
            self.exps = [] # specifit obama nerf
            self.face_rect = []
            self.lips_rect = []
            bg = cv2.imread(os.path.join(self.root_path, "bc.jpg"), cv2.IMREAD_UNCHANGED) # [H, W, 3]
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, 'gt_imgs', str(f['img_id']) + '.jpg')
                mask_path = os.path.join(self.root_path, 'parsing', str(f['img_id']) + '.png')
                lms = np.loadtxt(os.path.join(self.root_path, 'ori_imgs', str(f['img_id']) + '.lms')) # [68, 2]

                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA)

                # 不会出现的情况，当downscale 始终设置为1时
                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                
                m = cv2.inRange(mask, np.array([0,0,250]), np.array([0,0,255]))
                img = cv2.bitwise_and(image, image, mask=m)
                inv_m = cv2.bitwise_not(m)
                b = cv2.bitwise_and(bg, bg, mask=inv_m)
                img = cv2.bitwise_or(img, b)

                img = img.astype(np.float32) / 255 # [H, W, 3/4]
                # cv2.imwrite("debug3.png", ((255 *image).astype(np.uint8))[:,:,::-1])
                # cv2.imwrite("debug.png", ((255 *img).astype(np.uint8))[:,:,::-1])

                m = m.astype(np.uint8) / 255
                self.poses.append(pose)
                self.images.append(img)
                self.masks.append(m)

                if self.exps is not None:
                    m = loadmat(os.path.join(self.root_path, 'coeffs', str(f['img_id']) + '_coeff.mat'))
                    exp = np.array(m['coeff_bs'], dtype=np.float32).flatten() # [46,]
                    self.exps.append(exp)

                    # load lms and extract face
                    lms = np.loadtxt(os.path.join(self.root_path, 'ori_imgs', str(f['img_id']) + '.lms')) # [68, 2]
                    xmin, xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max())
                    ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
                    self.face_rect.append([xmin, xmax, ymin, ymax])
                
                if self.opt.finetune_lips:
                    lips = slice(48, 60)
                    xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
                    ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

                    # padding to H == W
                    cx = (xmin + xmax) // 2
                    cy = (ymin + ymax) // 2

                    l = max(xmax - xmin, ymax - ymin) // 2
                    xmin = max(0, cx - l) # 越界的时候的光线数量就不够
                    xmax = min(self.H, cx + l)
                    ymin = max(0, cy - l)
                    ymax = min(self.W, cy + l)
                    self.lips_rect.append([xmin, xmax, ymin, ymax])

            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
            self.masks = torch.from_numpy(np.stack(self.masks, axis=0)) # [N, H, W, C]
        if self.exps is not None:
            self.exps = torch.from_numpy(np.stack(self.exps, axis=0)) # [N, 46]
            self.max_exp, _ = torch.max(self.exps, axis=0)
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
                self.masks = self.masks.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        elif "focal_len" in transform:
            fl_x = transform['focal_len']
            fl_y = transform['focal_len']
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        # process background
        bg_img = cv2.imread(os.path.join(self.root_path, "bc.jpg"), cv2.IMREAD_UNCHANGED) # [H, W, 3]
        if bg_img.shape[0] != self.H or bg_img.shape[1] != self.W:
            bg_img = cv2.resize(bg_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        bg_img = bg_img.astype(np.float32) / 255 # [H, W, 3/4]
        self.bg_img = bg_img
        self.bg_img = torch.from_numpy(self.bg_img)
        self.bg_img = self.bg_img.to(torch.half).to(self.device)
        # self.bg_img = None


    def collate(self, index):

        B = len(index) # a list of length 1

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        if self.training and self.opt.finetune_lips and self.opt.patch_size > 1:
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, patch_size=self.opt.patch_size, rect=self.lips_rect[index[0]])
        else:
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, patch_size=self.opt.patch_size, rect=None)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }
        if self.bg_img is not None:
            bg_img = self.bg_img.view(1, -1, 3).repeat(B, 1, 1).to(self.device)
            if self.training:
                bg_img = torch.gather(bg_img, 1, torch.stack(3 * [rays['inds']], -1)) # [B, N, 3]
            results['bg_color'] = bg_img
        else:
            results['bg_color'] = None


        if self.exps is not None:
            exps = self.exps[index].to(self.device) # [B, 46]
            results['weight'] = exps
            results['max_weight'] = self.max_exp

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]\
            masks = self.masks[index].to(self.device)
            # import pdb; pdb.set_trace()
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
                masks = torch.gather(masks.view(B, -1, 1), 1, torch.stack([rays['inds']], -1)) # [B, N]
            results['images'] = images
            results['mask'] = masks
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=self.opt.batch_size, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader