import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from nerf_obama.network import NeRFNetwork
from nerf_obama.utils import get_rays
import torch
import numpy as np
import cv2
from scipy.io import loadmat
import json
# from math import cos, sin
import tqdm
import imageio
''' 
Input a animation video
Generate the corresponding face
'''

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


class InferenceBSNerf():
    def __init__(self) -> None:
        self.checkpoint = 'result/h2/checkpoints/ngp_ep0023.pth'
        self.device = 'cuda'

        self.fl_x, self.fl_y = 1300, 1300
        self.cx, self.cy = 256, 256
        self.intrinsics = np.array([self.fl_x, self.fl_y, self.cx, self.cy]) 
        self.H, self.W = 512,512
        self.scale = 4

        self.pose = None
        self.exp = None
        self.loadminmax()
        self.loadbg()
        self.load()

    def loadminmax(self):
        self.max_coeff = np.load("/raid/xjd/torch-ngp/data/0002/max_coeffs.npy")[None, ...] # [46, ]
        self.min_coeff = np.load("/raid/xjd/torch-ngp/data/0002/min_coeffs.npy")[None, ...]
        print(self.max_coeff)

    def loadbg(self):
        bg_img = cv2.imread("data/0002/bc.jpg", cv2.IMREAD_UNCHANGED) # [H, W, 3]
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        bg_img = bg_img.astype(np.float32) / 255 # [H, W, 3/4]
        bg_img = torch.from_numpy(bg_img)
        bg_img = bg_img.view(1, -1, 3).repeat(1, 1, 1).to(self.device)
        self.bg_color = bg_img

    def render(self, pose, exp):
        self.poses = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        for i in range(exp.shape[1]):
            if exp[:,i] > self.max_coeff[:,i]:
                exp[:,i] = self.max_coeff[:,i]
            if exp[:,i] < self.min_coeff[:,i]:
                exp[:,i] = self.min_coeff[:,i]
        weight = torch.from_numpy(exp).to(self.device)

        rays = get_rays(self.poses, self.intrinsics, self.H, self.W, N=-1, error_map=None, patch_size=1, rect=None)
        rays_o = rays['rays_o']
        rays_d = rays['rays_d']

        outputs = self.model.render(rays_o, rays_d, weight=weight, staged=True, bg_color=self.bg_color, perturb=True)
        pred_rgb = outputs['image'].reshape(-1, self.H, self.W, 3)
        pred_depth = outputs['depth'].reshape(-1, self.H, self.W)
        pred = pred_rgb[0].detach().cpu().numpy()
        pred = (pred * 255).astype(np.uint8)

        return pred

    def load(self):
        self.model = NeRFNetwork(
            encoding="hashgrid",
            bound=1,
            cuda_ray=True,
            density_scale=1,
            min_near=0.2,
            density_thresh=10,
            device=self.device
        )
        self.model.to(self.device)

        checkpoint_dict = torch.load(self.checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint_dict['model'], strict=False)


if __name__ == '__main__':

    model = InferenceBSNerf()

    path = "/raid/xjd/torch-ngp/data/0004"
    save_path = "exp_test"
    with open(os.path.join(path, 'transforms_test.json'), 'r') as f:
        transform = json.load(f)
    # read images
    frames = transform["frames"]
    vimg = []
    for f in tqdm.tqdm(frames):
        image = cv2.imread(os.path.join(path, 'gt_imgs', str(f['img_id']) + '.jpg'))[:,:,::-1]
        pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
        pose[3,:] = np.array([-0.33, -0.18, 0.69, 1])
        # print(pose)
        pose = nerf_matrix_to_ngp(pose, scale=4, offset=[0,0,0])
        m = loadmat(os.path.join(path, 'coeffs', str(f['img_id']) + '_coeff.mat'))
        exp = np.array(m['coeff_bs'], dtype=np.float32).flatten()[None, ...] # [46,]
        pred = model.render(pose, exp)
        cv2.imwrite("exp_test/exp_source.png", cv2.cvtColor(np.hstack((image, pred)), cv2.COLOR_BGR2RGB))
        vimg.append(np.hstack((image, pred)))

    imageio.mimwrite(os.path.join(save_path, f'{42}_rgb.mp4'), vimg, fps=25, quality=8, macro_block_size=1)


