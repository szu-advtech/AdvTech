import h5py
import cv2
import numpy as np


save_path = r'C:\Users\VCC\Desktop\CameraViewPretrain\CameraViewPretrain\data\density_maps/camera3/'

with h5py.File(r"C:\Users\VCC\Desktop\CameraViewPretrain\CameraViewPretrain\Street_view3_dmap_10.h5") as f:
    img = f['density_maps']
    for i in range(300):
        cur = img[i]
        name = str(636 + i * 2)
        if len(name) == 3:
            name = '0' + name
        img_name = name + '.npy'
        np.save(save_path + img_name, cur)


