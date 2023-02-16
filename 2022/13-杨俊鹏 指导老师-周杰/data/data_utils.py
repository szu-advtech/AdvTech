import torch
import numpy as np

class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()

def angle_axis(angle: float, axis: np.ndarray):
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()    

class PointcloudRotatebyAngle(object):
    def __init__(self, rotation_angle = 0.0):
        self.rotation_angle = rotation_angle

    def __call__(self, pc):
        normals = pc.size(2) > 3
        bsize = pc.size()[0]
        for i in range(bsize):
            cosval = np.cos(self.rotation_angle)
            sinval = np.sin(self.rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            rotation_matrix = torch.from_numpy(rotation_matrix).float().cuda()
            
            cur_pc = pc[i, :, :]
            if not normals:
                cur_pc = cur_pc @ rotation_matrix
            else:
                pc_xyz = cur_pc[:, 0:3]
                pc_normals = cur_pc[:, 3:]
                cur_pc[:, 0:3] = pc_xyz @ rotation_matrix
                cur_pc[:, 3:] = pc_normals @ rotation_matrix
                
            pc[i, :, :] = cur_pc
            
        return pc

class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data
            
        return pc

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc
        
class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
            
        return pc
        
class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()
            
        return pc

class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                # cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
                cur_pc[drop_idx.tolist(), :] = cur_pc[0, :].repeat(len(drop_idx), 1)
                pc[i, :, :] = cur_pc

        return pc

# my rotation augmentation:
class PointcloudRotatebyRandomAngle(object):
    def __init__(self, rotation_angle = 0.0, axis=np.array([0.0, 1.0, 0.0])):
        self.rotation_angle = rotation_angle 
        self.axis = axis

    def __call__(self, pc):
        normals = pc.size(2) > 3
        bsize = pc.size()[0]
        # ratio = np.random.uniform(-1,1)  
        ratio_list = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        ratio = np.random.choice(ratio_list)
        random_angle = self.rotation_angle * ratio
        for i in range(bsize):
            rotation_matrix = angle_axis(random_angle, self.axis)
            rotation_matrix = rotation_matrix.cuda()
            
            cur_pc = pc[i, :, :]
            if not normals:
                cur_pc = cur_pc @ rotation_matrix
            else:
                pc_xyz = cur_pc[:, 0:3]
                pc_normals = cur_pc[:, 3:]
                cur_pc[:, 0:3] = pc_xyz @ rotation_matrix
                cur_pc[:, 3:] = pc_normals @ rotation_matrix
                
            pc[i, :, :] = cur_pc
            
        return pc

# add noise augmentation:
class PointcloudJitterAxisZ(object):
    def __init__(self, std=0.039, clip=0.156):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1)).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 2] += jittered_data
            
        return pc

# my occlusion augmentation:
class PointcloudManmadeOcclusion(object):
    def __init__(self):
        super().__init__()
        # self.direction = direction
        # self.dim = dim
        
    def __call__(self, pc):
        bsize = pc.size(0)
        for i in range(bsize):
            cur_pc = pc[i,:,:]
            # centroid =  torch.mean(cur_pc, dim=0, keepdim=True)
            # centroid[0, self.dim] = torch.max(cur_pc[:, self.dim])
            N = cur_pc.size()[0]
            # if self.direction == 'left':
            #     crop_index = [ i for i in np.arange(0, N) if centroid[0,0]-0.1 > cur_pc[i,0]]
            # elif self.direction == 'right':
            #     crop_index = [ i for i in np.arange(0, N) if centroid[0,0]+0.1 < cur_pc[i,0]]
            
            rand_idx = np.random.choice(np.arange(0, N))
            crop_index = [j for j in np.arange(0,N) if torch.sum(torch.pow(cur_pc[rand_idx,:]-cur_pc[j, :], 2)) < 0.1]
            maintain_index = set([i for i in np.arange(0, N)]) - set(crop_index)
            maintain_index = list(maintain_index)
            cur_pc[crop_index, :] = cur_pc[maintain_index[0], :].repeat(len(crop_index), 1)

            pc[i,:,:] = cur_pc
        
        return pc