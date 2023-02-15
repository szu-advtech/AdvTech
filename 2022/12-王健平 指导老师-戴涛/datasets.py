import h5py
import numpy as np
from torch.utils.data import Dataset


# 训练时，用低分辨率 lr
class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    # 支持下标索引访问数据集
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # 返回 0-1
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)
    # 应该提供数据集的大小(容量)
    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

# 评估时，用高分辨率 hr
class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
