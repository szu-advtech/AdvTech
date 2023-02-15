import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision import transforms


def conv_process(img, pad=0, stride=4, filter_size=4, dim_ordering='tf'):

    assert img.ndim == 3
    assert stride == filter_size
    assert dim_ordering in ['th', 'tf']
    if dim_ordering == 'th':
        hy_rows = img.shape[1]
        wx_cols = img.shape[2]
        n_channel = img.shape[0]
    elif dim_ordering == 'tf':
        hy_rows = img.shape[0]
        wx_cols = img.shape[1]
        n_channel = img.shape[2]
    assert hy_rows % filter_size == 0
    assert wx_cols % filter_size == 0
    assert n_channel in [1]

    range_y = range(0, hy_rows + 2 * pad - filter_size + 1, stride)

    range_x = range(0, wx_cols + 2 * pad - filter_size + 1, stride)

    output_rows = len(range_y)
    output_cols = len(range_x)

    if dim_ordering == 'th':
        result = np.zeros((n_channel, output_rows, output_cols), dtype=np.single)
    elif dim_ordering == 'tf':
        result = np.zeros((output_rows, output_cols, n_channel), dtype=np.single)
    for index in range(n_channel):
        if dim_ordering == 'th':
            if pad > 0:
                new_data = np.zeros(
                    [hy_rows + 2 * pad, wx_cols + 2 * pad], dtype=np.single)
                new_data[pad:pad + hy_rows, pad:pad + wx_cols] = img[index, ...]
            else:
                new_data = img[index, ...]

            y_ind = 0
            for y in range_y:
                x_ind = 0
                for x in range_x:
                    result[index, y_ind, x_ind] = new_data[y:y + filter_size, x:x + filter_size].sum()
                    x_ind += 1
                y_ind += 1
        elif dim_ordering == 'tf':
            if pad > 0:
                new_data = np.zeros(
                    [hy_rows + 2 * pad, wx_cols + 2 * pad], dtype=np.single)
                new_data[pad:pad + hy_rows, pad:pad + wx_cols] = img[..., index]
            else:
                new_data = img[..., index]

            y_ind = 0
            for y in range_y:
                x_ind = 0
                for x in range_x:
                    result[y_ind, x_ind, index] = new_data[y:y + filter_size, x:x + filter_size].sum()
                    x_ind += 1
                y_ind += 1
    return result



def get_img_list(file):
    with open(file, 'r') as f:
        img_list = [item[:-1] for item in f]
    return img_list


class M_v_dataset(Dataset):
    def __init__(self, view, data_dir=r"/home/kaiyi/camera_view_pretrain/CameraViewPretrain/data//"):
        self.view = view
        self.data_dir = data_dir
        self.im_li_d = self.data_dir + 'img_train.txt'
        self.v1_dir = data_dir + 'camera1/train'
        self.v2_dir = data_dir + 'camera2/train'
        self.v3_dir = data_dir + 'camera3/train'
        self.density_dir = self.data_dir + 'density_maps/'
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize([380, 676])
        ])

    def __len__(self):
        return len(get_img_list(self.im_li_d))

    def __getitem__(self, idx):
        img_list = get_img_list(self.im_li_d)
        img_name = img_list[idx % len(img_list)]

        img_v1_p = os.path.join(self.v1_dir, img_name)
        img_v2_p = os.path.join(self.v2_dir, img_name)
        img_v3_p = os.path.join(self.v3_dir, img_name)
        img_v1 = Image.open(img_v1_p).convert('RGB')
        img_v2 = Image.open(img_v2_p).convert('RGB')
        img_v3 = Image.open(img_v3_p).convert('RGB')
        print(img_v3_p)

        density_maps_c1 = np.load(os.path.join(self.density_dir + 'camera1', img_name.replace('jpg', 'npy')))
        density_maps_c1 = conv_process(density_maps_c1) * 1000

        density_maps_c2 = np.load(os.path.join(self.density_dir + 'camera2', img_name.replace('jpg', 'npy')))
        density_maps_c2 = conv_process(density_maps_c2) * 1000

        density_maps_c3 = np.load(os.path.join(self.density_dir + 'camera3', img_name.replace('jpg', 'npy')))
        density_maps_c3 = conv_process(density_maps_c3) * 1000

        if self.view == 1:
            return self.trans(img_v1), torch.from_numpy(density_maps_c1.copy()).float()
        elif self.view == 2:
            return self.trans(img_v2), torch.from_numpy(density_maps_c2.copy()).float()
        else:
            return self.trans(img_v3), torch.from_numpy(density_maps_c3.copy()).float()
