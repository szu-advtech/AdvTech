import numpy as np
import torch
import h5py
from sklearn.neighbors import KDTree
from src.utils.ndc_utils import *
# from src.utils.ndc_utils import read_data, read_and_augment_data_undc2, read_data_input_only, read_data2,read_and_augment_data_undc3

MOTION_DIR = "/home/fubao/dev/InterRecon/motion"
STATE_DIR = "/home/fubao/dev/InterRecon/state/cate_shape"
CATE_LIST = ['dishwasher','kitchenpot','microwave','oven','refrige','storage','table','trashcan']

class ABC_pc_hdf5(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        input_point_num,
        output_grid_size,
        KNN_num,
        pooling_radius,
        phase,
        input_only=False
    ):
        self.data_dir = data_dir
        self.input_point_num = input_point_num
        self.output_grid_size = output_grid_size
        self.KNN_num = KNN_num
        self.pooling_radius = pooling_radius
        self.input_only = input_only
        self.pcs_txt_path_list = []
        self.phase = "phase"
        
        name_list2_dict = {}
        valid_list_dict = {}
        partnum_list_dict = {}
        refpart_list_dict = {}
        movpart_list_dict = {}
        part_idx_list_dict = {}
        idx_dict_dict = {}
        
        for cate in CATE_LIST:
            h5_file = h5py.File(f"{MOTION_DIR}/{cate}_motion.h5", 'r')
            name_list2_dict[cate] = h5_file["name"][:]
            name_list2_dict[cate] = [name.decode('utf-8') for name in name_list2_dict[cate]]
            valid_list_dict[cate] = h5_file["valid"][:].astype(np.int32)
            partnum_list_dict[cate] = h5_file["partnum"][:].astype(np.int32)
            refpart_list_dict[cate] = h5_file["refpart"][:].astype(np.int32)
            movpart_list_dict[cate] = h5_file["movpart"][:].astype(np.int32)
            part_idx_list_dict[cate] = [[0 for i in range(26)] for partnum in partnum_list_dict[cate]]
            idx_dict_dict[cate] = {}
            len_data = int(len(name_list2_dict[cate])/13)
            for i in range(len_data):
                name = name_list2_dict[cate][13*i]
                shape_id = name.split("_")[0]
                partnum = partnum_list_dict[cate][13*i]
                movpart = movpart_list_dict[cate][13*i]
                for j in range(partnum-1):
                    part_idx_list_dict[cate][13*i][movpart[j]] = j
                idx_dict_dict[cate][shape_id] = 13*i
        
        data_list = []
        state_file_path = f"{STATE_DIR}/{phase}.txt"
        with open(state_file_path,"r") as fp:
            for line in fp.readlines():
                data_list.append(line.strip())    
                   
        # print(data_list)       
        self.hdf5_names = []
        with open(f"{data_dir}/my_obj_list.txt", "r") as fp:
            for name in fp.readlines():
                name = name.strip()
                cate_shape_id = name.split("_")[0]
                
                if cate_shape_id in data_list:
                    self.hdf5_names.append(name)
        
        hdf5_names2 = []
        for name in self.hdf5_names:
            cate, file_name = name.split("/")
            shape_id, sid, pid, fid, prid = file_name.split("_")
            pr_id = int(prid)
            idx = idx_dict_dict[cate][shape_id]
            r_id = refpart_list_dict[cate][idx][0]
            if pr_id == r_id and fid == 'f00':
                hdf5_names2.append(name)
                self.pcs_txt_path_list.append(f"{cate}/{shape_id}_base")
            if pr_id != r_id and fid == 'f09':
                hdf5_names2.append(name)
                self.pcs_txt_path_list.append(f"{cate}/{shape_id}_p{prid}")
        self.hdf5_names = hdf5_names2
        
        if phase == "train":
            # ******* why
            # separate 32 and 64
            temp_hdf5_names = []
            temp_pcs_paths = []
            temp_hdf5_gridsizes = []
            index = 0
            for name in self.hdf5_names:
                for grid_size in [32, 64]:
                    temp_hdf5_names.append(name)
                    temp_pcs_paths.append(self.pcs_txt_path_list[index])
                    temp_hdf5_gridsizes.append(grid_size)
                index += 1
            # ******* end why
            self.hdf5_names = temp_hdf5_names
            self.pcs_txt_path_list = temp_pcs_paths
            self.hdf5_gridsizes = temp_hdf5_gridsizes
        else:
            self.hdf5_gridsizes = [self.output_grid_size]*len(self.hdf5_names)

    def __len__(self):
        return len(self.hdf5_names) 

    def __getitem__(self, index):
        gt_output_bool = torch.randn(1, 1)
        gt_output_float = torch.randn(1, 1)
        gt_output_float_mask = torch.randn(1, 1)

        hdf5_dir = f"{self.data_dir}/{self.hdf5_names[index]}.hdf5"
        cate, file_name = self.hdf5_names[index].split("/")
        shape_id,sid,pid,fid,prid = file_name.split("_")
        name_info = f"{cate}_{shape_id}"
        # pcs_txt_path = f"{PC_PATH}/{self.pcs_txt_path_list[index]}.txt"
        grid_size = self.hdf5_gridsizes[index]
        if self.phase == "train":
            # gt_output_bool_, gt_output_float_, gt_input_ = read_and_augment_data_undc2(
            #     hdf5_dir, pcs_txt_path, grid_size, "pointcloud", True, True, aug_permutation=True, aug_reversal=True, aug_inversion=False)
            gt_output_bool_, gt_output_float_, gt_input_ = read_and_augment_data_undc(
                hdf5_dir, grid_size, "pointcloud", True, True, aug_permutation=True, aug_reversal=True, aug_inversion=False)
        else:
            # ? what is input_only
            if self.input_only:
                gt_output_bool_, gt_output_float_, gt_input_ = read_data_input_only(
                    hdf5_dir, grid_size, "pointcloud", True, True, is_undc=True)
            else:
                gt_output_bool_, gt_output_float_, gt_input_ = read_data(
                    hdf5_dir, grid_size, "pointcloud", True, True,is_undc=True)

        if self.phase == "train":
            # augment input point number depending on the grid size
            # grid   ideal?  range
            # 32     2048    1024-4096
            # 64     8192    4096-16384
            np.random.shuffle(gt_input_)
            if grid_size == 32:
                count = np.random.randint(1024, 4096)
            elif grid_size == 64:
                count = np.random.randint(4096, 16384)
            gt_input_ = gt_input_[:count]
        else:
            gt_input_ = gt_input_[:self.input_point_num]
        gt_input_ = np.ascontiguousarray(gt_input_)


        pc_xyz = gt_input_
        kd_tree = KDTree(pc_xyz, leaf_size=8)
        pc_KNN_idx = kd_tree.query(pc_xyz, k=self.KNN_num, return_distance=False)

        pc_KNN_idx = np.reshape(pc_KNN_idx, [-1])
        pc_KNN_xyz = pc_xyz[pc_KNN_idx]
        pc_KNN_xyz = np.reshape(pc_KNN_xyz, [len(pc_xyz), self.KNN_num, 3]) - np.reshape(pc_xyz, [len(pc_xyz), 1, 3])
        pc_KNN_xyz = np.reshape(pc_KNN_xyz, [len(pc_xyz)*self.KNN_num, 3])
        # this will be used to group point features

        # consider all grid cells within range to the point cloud
        pc_xyz_int = np.floor(pc_xyz).astype(np.int32)
        pc_xyz_int = np.clip(pc_xyz_int, 0, grid_size)
        tmp_grid = np.zeros([grid_size+1, grid_size+1, grid_size+1], np.uint8)
        tmp_grid[pc_xyz_int[:, 0], pc_xyz_int[:, 1], pc_xyz_int[:, 2]] = 1
        for ite in range(self.pooling_radius):
            tmp_mask = np.copy(tmp_grid[1:-1, 1:-1, 1:-1])
            for i in range(3):
                for j in range(3): 
                    for k in range(3):
                        tmp_grid[i:grid_size-1+i, j:grid_size-1+j, k:grid_size-1 +k] = tmp_mask | tmp_grid[i:grid_size-1+i, j:grid_size-1+j, k:grid_size-1+k]

        voxel_x, voxel_y, voxel_z = np.nonzero(tmp_grid)
        voxel_xyz = np.concatenate([np.reshape(voxel_x, [-1, 1]), np.reshape(voxel_y, [-1, 1]), np.reshape(voxel_z, [-1, 1])], 1)
        voxel_xyz = voxel_xyz.astype(np.float32)+0.5
        voxel_xyz_int = np.floor(voxel_xyz).astype(np.int64)

        voxel_KNN_idx = kd_tree.query(voxel_xyz, k=self.KNN_num, return_distance=False)
        voxel_KNN_idx = np.reshape(voxel_KNN_idx, [-1])
        voxel_KNN_xyz = pc_xyz[voxel_KNN_idx]
        voxel_KNN_xyz = np.reshape(voxel_KNN_xyz, [len(voxel_xyz), self.KNN_num, 3]) - np.reshape(voxel_xyz, [len(voxel_xyz), 1, 3])
        voxel_KNN_xyz = np.reshape(voxel_KNN_xyz, [len(voxel_xyz)*self.KNN_num, 3])

        gt_output_bool = gt_output_bool_[voxel_xyz_int[:, 0], voxel_xyz_int[:, 1], voxel_xyz_int[:, 2]]
        gt_output_bool = np.ascontiguousarray(gt_output_bool, np.float32)

        gt_output_float = gt_output_float_[voxel_xyz_int[:, 0], voxel_xyz_int[:, 1], voxel_xyz_int[:, 2]]
        gt_output_float = np.ascontiguousarray(gt_output_float, np.float32)
        gt_output_float_mask = (gt_output_float >= 0).astype(np.float32)

        return pc_KNN_idx, pc_KNN_xyz, voxel_xyz_int, voxel_KNN_idx, voxel_KNN_xyz, gt_output_bool, gt_output_float, gt_output_float_mask,gt_input_,name_info


class ABC_npc_hdf5(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        input_point_num,
        output_grid_size,
        KNN_num,
        pooling_radius,
        train,
        input_only=False
    ):
        self.data_dir = data_dir
        self.input_point_num = input_point_num
        self.output_grid_size = output_grid_size
        self.KNN_num = KNN_num
        self.pooling_radius = pooling_radius
        self.train = train
        self.input_only = input_only

        with open(f"{data_dir}/abc_obj_list.txt", "r") as fp:
            self.hdf5_names = [name.strip() for name in fp.readlines()]

        if self.train:
            self.hdf5_names = self.hdf5_names[:int(len(self.hdf5_names)*0.8)]
            #augmented data
            temp_hdf5_names = []
            temp_hdf5_shape_scale = []
            for t in range(len(self.hdf5_names)):
                for s in [10,9,8,7,6,5]:
                    for i in [0,1]:
                        for j in [0,1]:
                            for k in [0,1]:
                                newname = self.hdf5_names[t]+"_"+str(s)+"_"+str(i)+"_"+str(j)+"_"+str(k)
                                temp_hdf5_names.append(newname)
                                temp_hdf5_shape_scale.append(s)
            self.hdf5_names = temp_hdf5_names
            self.hdf5_shape_scale = temp_hdf5_shape_scale
        else:
            self.hdf5_names = self.hdf5_names[int(len(self.hdf5_names)*0.8):]
            self.hdf5_shape_scale = [10]*len(self.hdf5_names)

    def __len__(self):
        return len(self.hdf5_names) 

    def __getitem__(self, index):
        gt_output_bool = torch.randn(1, 1)
        gt_output_float = torch.randn(1, 1)
        gt_output_float_mask = torch.randn(1, 1)

        hdf5_dir = f"{self.data_dir}/{self.hdf5_names[index]}.hdf5"
        grid_size = self.output_grid_size
        shape_scale = self.hdf5_shape_scale[index]

        if self.train:
            gt_output_bool_, gt_output_float_, gt_input_ = read_and_augment_data_undc(
                hdf5_dir, grid_size, "noisypc", True, True, aug_permutation=True, aug_reversal=True, aug_inversion=False)
        else:
            # ? what is input_only
            if self.input_only:
                gt_output_bool_, gt_output_float_, gt_input_ = read_data_input_only(
                    hdf5_dir, grid_size, "noisypc", True, True, is_undc=True)
            else:
                gt_output_bool_, gt_output_float_, gt_input_ = read_data(
                    hdf5_dir, grid_size, "noisypc", True, True, is_undc=True)

        if self.train:
            #augment input point number depending on the shape scale
            #grid   ideal?  range
            #64     16384    8192-32768
            np.random.shuffle(gt_input_)
            rand_int_s = int(8192*(shape_scale/10.0)**2)
            rand_int_t = int(32768*(shape_scale/10.0)**2)
            count = np.random.randint(rand_int_s,rand_int_t)
            gt_input_ = gt_input_[:count]
        else:
            gt_input_ = gt_input_[:self.input_point_num]
        gt_input_ = np.ascontiguousarray(gt_input_)
        
        if not self.train:
            np.random.seed(0)
        gt_input_ = gt_input_ + np.random.randn(gt_input_.shape[0],gt_input_.shape[1]).astype(np.float32)*0.5

        pc_xyz = gt_input_
        kd_tree = KDTree(pc_xyz, leaf_size=8)
        pc_KNN_idx = kd_tree.query(pc_xyz, k=self.KNN_num, return_distance=False)

        pc_KNN_idx = np.reshape(pc_KNN_idx, [-1])
        pc_KNN_xyz = pc_xyz[pc_KNN_idx]
        pc_KNN_xyz = np.reshape(pc_KNN_xyz, [len(pc_xyz), self.KNN_num, 3]) - np.reshape(pc_xyz, [len(pc_xyz), 1, 3])
        pc_KNN_xyz = np.reshape(pc_KNN_xyz, [len(pc_xyz)*self.KNN_num, 3])
        # this will be used to group point features

        # consider all grid cells within range to the point cloud
        pc_xyz_int = np.floor(pc_xyz).astype(np.int32)
        pc_xyz_int = np.clip(pc_xyz_int, 0, grid_size)
        tmp_grid = np.zeros([grid_size+1, grid_size+1, grid_size+1], np.uint8)
        tmp_grid[pc_xyz_int[:, 0], pc_xyz_int[:, 1], pc_xyz_int[:, 2]] = 1
        for ite in range(self.pooling_radius):
            tmp_mask = np.copy(tmp_grid[1:-1, 1:-1, 1:-1])
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        tmp_grid[i:grid_size-1+i, j:grid_size-1+j, k:grid_size-1 +
                                 k] = tmp_mask | tmp_grid[i:grid_size-1+i, j:grid_size-1+j, k:grid_size-1+k]

        voxel_x, voxel_y, voxel_z = np.nonzero(tmp_grid)
        voxel_xyz = np.concatenate([np.reshape(
            voxel_x, [-1, 1]), np.reshape(voxel_y, [-1, 1]), np.reshape(voxel_z, [-1, 1])], 1)
        voxel_xyz = voxel_xyz.astype(np.float32)+0.5
        voxel_xyz_int = np.floor(voxel_xyz).astype(np.int64)

        voxel_KNN_idx = kd_tree.query(voxel_xyz, k=self.KNN_num, return_distance=False)
        voxel_KNN_idx = np.reshape(voxel_KNN_idx, [-1])
        voxel_KNN_xyz = pc_xyz[voxel_KNN_idx]
        voxel_KNN_xyz = np.reshape(voxel_KNN_xyz, [len(voxel_xyz), self.KNN_num, 3]) - np.reshape(voxel_xyz, [len(voxel_xyz), 1, 3])
        voxel_KNN_xyz = np.reshape(voxel_KNN_xyz, [len(voxel_xyz)*self.KNN_num, 3])

        gt_output_bool = gt_output_bool_[voxel_xyz_int[:, 0], voxel_xyz_int[:, 1], voxel_xyz_int[:, 2]]
        gt_output_bool = np.ascontiguousarray(gt_output_bool, np.float32)

        gt_output_float = gt_output_float_[voxel_xyz_int[:, 0], voxel_xyz_int[:, 1], voxel_xyz_int[:, 2]]
        gt_output_float = np.ascontiguousarray(gt_output_float, np.float32)
        gt_output_float_mask = (gt_output_float >= 0).astype(np.float32)

        return pc_KNN_idx, pc_KNN_xyz, voxel_xyz_int, voxel_KNN_idx, voxel_KNN_xyz, gt_output_bool, gt_output_float, gt_output_float_mask
