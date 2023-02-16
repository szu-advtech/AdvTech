"""
@Name: val.py
@Auth: SniperIN_IKBear
@Date: 2022/12/8-19:59
@Desc: 
@Ver : 0.0.0
"""
import hdf5storage
from tqdm import tqdm

import Model
import h5py
import numpy as np
import torch

hsi_path = '/home2/szx/Dataset/TreeDetection/DHIF/Test3/HSI/10240x2560.mat'
rgb_path = '/home2/szx/Dataset/TreeDetection/DHIF/Test3/RGB/10240x2560.mat'

img_hsi = np.array(h5py.File(hsi_path)['hsi'])[:, :2560, :]
img_rgb = np.array(h5py.File(rgb_path)['rgb'])[:, :2560, :]


img_rgb = torch.from_numpy(img_rgb).unsqueeze(0)
img_hsi = torch.from_numpy(img_hsi).unsqueeze(0)
assert len(img_rgb.shape) == 4
# for i in tqdm(range(1)):
model = Model.HSI_Fusion(Ch=53, stages=4, sf = 8)
model.load_state_dict(torch.load("./Checkpoint/f8/Model/model_022.pth",map_location=lambda storage, loc: storage).module.state_dict())
model = model.eval()
print('model complete')
with torch.no_grad():
    out = model(img_rgb.cpu(), img_hsi.cpu())
    result = out
    result = result.clamp(min=0., max=1.)
#
res = result.cpu().permute(2, 3, 1, 0).squeeze(3).numpy()
save_path = './Result/ssr/resx8.mat'
print("write data.....")
hdf5storage.savemat(save_path, {'res': res}, do_compression=True, format='7.3')
print(res.shape)
print("write data over!  .....")


