import os
from tqdm import tqdm
import cv2
import imageio
from .base import BaseDataset
from utils.ray_utils import *
from einops import repeat

from utils.colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary

def read_image(img_path, img_wh, blend_a=True):
    img = imageio.imread(img_path).astype(np.float32) / 255.0
    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            img = img[..., :3] * img[..., -1:]
    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')

    return img


class RffrDataset(BaseDataset):
    def __init__(self, root_dir, split='train', **kwargs, ):
        self.downsample = kwargs.get("downsample")
        super().__init__(root_dir, split, self.downsample)
        self.batch_size = kwargs.get("batch_size")
        self.read_intrinsics()
        self.dataset_name = "rffr"
        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)


    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height * self.downsample)
        w = int(camdata[1].width * self.downsample)
        self.img_wh = (w, h)
        self.pix_idxs = repeat(torch.cat([torch.arange(4) + i * self.img_wh[0] for i in range(4)]), "n->m n",
                               m=int(self.batch_size / (4 ** 2)))
        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0] * self.downsample
            cx = camdata[1].params[1] * self.downsample
            cy = camdata[1].params[2] * self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0] * self.downsample
            fy = camdata[1].params[1] * self.downsample
            cx = camdata[1].params[2] * self.downsample
            cy = camdata[1].params[3] * self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        folder = 'images'
        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]

        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3]  # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d])  # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= scale
        self.pts3d /= scale

        self.rays = []
        self.mask = []
        self.all_masks_valid = []

        if split == "train":
            with open(os.path.join(self.root_dir, "train.txt"), "r") as file:
                file_names = file.readlines()
                file_names = [os.path.join(self.root_dir, "images/", file_name.split("\n")[0]) for file_name in
                              file_names]
                idx = [index for index, value in enumerate(img_paths) if value in file_names]
                img_paths = [img_paths[index] for index in idx]
                self.poses = np.array([x for i, x in enumerate(self.poses) if i in idx])
        elif split == "test":
            with open(os.path.join(self.root_dir, "val.txt"), "r") as file:
                file_names = file.readlines()
                file_names = [os.path.join(self.root_dir, "images/", file_name.split("\n")[0]) for file_name in
                              file_names]
                idx = [index for index, value in enumerate(img_paths) if value in file_names]
                img_paths = [img_paths[index] for index in idx]
                self.poses = np.array([x for i, x in enumerate(self.poses) if i in idx])


        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            buf = []  # buffer for ray attributes: rgb, etc
            img = read_image(img_path, self.img_wh, blend_a=False)
            img = torch.FloatTensor(img)
            buf += [img]
            self.rays += [torch.cat(buf, 1)]
            mask_path = os.path.join(self.root_dir, 'refl_masks', img_path.split('.')[0] + '.png')
            if os.path.exists(mask_path):
                buf = []  # buffer for ray attributes: rgb, etc
                mask = read_image(mask_path, self.img_wh, blend_a=False)
                mask = torch.FloatTensor(mask)
                mask = mask[:, :, 3]
                mask[mask > 0] = 1
                buf += [mask]
                self.mask += [torch.cat(buf, 1)]
                self.all_masks_valid += [torch.ones_like(mask)]
            else:
                mask = torch.zeros_like(img)[:, [0]]
                self.mask += [mask]
                self.all_masks_valid += [torch.zeros_like(mask)]

        self.rays = torch.stack(self.rays)  # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses)  # (N_images, 3, 4)
        self.mask = torch.stack(self.mask)  # (N_images, hw, ?)
        self.all_masks_valid = torch.stack(self.all_masks_valid)  # (N_images, hw, ?)
