if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/d/pancheng/Project/IDR-Jittor/code')
    
    
import jittor as jt
import os
import numpy as np
from utils import rend_util
from jittor.dataset.dataset import Dataset
import utils.general as utils

class SceneDataset(Dataset):
    def __init__(self,
                 train_cameras,
                 data_dir,
                 img_res,
                 scan_id=65,
                 cam_file=None,
                 batch_size=1,
                 shuffle=True):
        super().__init__()
        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))
        self.batch_size = batch_size
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.shuffle = shuffle
        # self.batch_size = batch_size
        assert os.path.exists(self.instance_dir), "Data directory is empty"
        
        self.train_cameras = train_cameras
        self.sampling_idx = None
        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        mask_dir = '{0}/mask'.format(self.instance_dir)
        mask_paths = sorted(utils.glob_imgs(mask_dir))
        
        self.n_images = len(image_paths)
        
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        if cam_file is not None:
            self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        
        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(jt.array(intrinsics).float())
            self.pose_all.append(jt.array(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(jt.array(rgb).float())

        self.object_masks = []
        for path in mask_paths:
            object_mask = rend_util.load_mask(path)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(jt.array(object_mask).bool())
            
        # print("self.rgb_images.shape:    ", self.rgb_images[0].shape)
        # print("self.object_masks.shape:    ", self.object_masks[0].shape)
        
    def __getitem__(self, index):
        # index = 1
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = jt.array(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        
        sample = {
            "object_mask": self.object_masks[index],
            "uv": uv,
            "intrinsics": self.intrinsics_all[index],
        }
        
        ground_truth = {
            "rgb": self.rgb_images[index]
        }
        
        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[index][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[index][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[index]

        return index, sample, ground_truth
        
    def __len__(self):
        return self.n_images
    
    def collate_batch(self, batch_list):
        batch_list = zip(*batch_list)
        # print("len(batch_list):   ", len(list(batch_list)))
        all_parsed = []
        cnt = 0
        for entry in batch_list:
            # cnt += 1
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    cnt += 1
                    ret[k] = jt.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(entry)
            # print("cnnnnnnnnnnt = ", cnt)
        
        # print("len(all_parsed):         ", len(all_parsed[0]), len(all_parsed[1]), len(all_parsed[2]))
        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
            print("NoneNOne")
        else:
            self.sampling_idx = jt.misc.randperm(self.total_pixels)[:sampling_size]
            
    
    def get_pose_init(self):
        # get noisy initializations obtained with the linear method
        cam_file = '{0}/cameras_linear_init.npz'.format(self.instance_dir)
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        init_pose = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            init_pose.append(pose)
        init_pose = jt.concat([jt.Var(pose).float().unsqueeze(0) for pose in init_pose], 0).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = jt.concat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat
# dataloader = SceneDataset(True, 'DTU', [1200, 1600], 65, batch_size=3, shuffle=True)
# dataloader.change_sampling_idx(2048)
# for data_index, (idx, model_input, ground_truth) in enumerate(dataloader):
#     print("data_index: ", data_index)
#     print("idx: ", idx)
#     print("object_mask: ", model_input['object_mask'].size())
#     print("uv: ", model_input['uv'].size())
#     print("intrinsics: ", model_input['intrinsics'].size())


# object_mask:  [3,1920000,]
# uv:  [3,1920000,2,]
# intrinsics:  [3,4,4,]
# data_index:  15
# idx:  [41, 18, 13]

# object_mask:  [3,2048,]
# uv:  [3,2048,2,]
# intrinsics:  [3,4,4,]
# data_index:  15
# idx:  [41, 18, 13]
