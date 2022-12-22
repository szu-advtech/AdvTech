# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/15 19:13
 @name: 
 @desc:
"""
import glob
import platform
from tqdm import tqdm
from data_load.read_mat import read_eeg_mat, read_locs_mat
from joblib import Parallel, delayed
import pickle
import numpy as np
from preprocessing.aep import azim_proj, gen_images
import einops

parallel_jobs = 1
locs = None


def thread_read_write(x, y, pkl_filename):
    """Writes and dumps the processed pkl file for each stimulus(or called subject).
    [time, channels=127], y
    """
    with open(pkl_filename + '.pkl', 'wb') as file:
        pickle.dump(x, file)
        pickle.dump(y, file)


def go_through(filenames, pkl_path):
    for f in tqdm(filenames, desc=' Total', position=0, leave=True, colour='YELLOW', ncols=80):
        eeg, y = read_eeg_mat(f)  # [n_samples=5184, t_length=32, channels=62]

        # -----------------
        samples, time, channels = np.shape(eeg)
        eeg = einops.rearrange(eeg, 'n t c -> (n t) c', n=samples, t=time, c=channels)

        locs_2d = [azim_proj(e) for e in locs]

        imgs = gen_images(locs=np.array(locs_2d),  # [samples*time, colors, W, H]
                          features=eeg,
                          n_gridpoints=32,
                          normalize=True).squeeze()

        imgs = einops.rearrange(imgs, '(n t) w h -> n t w h', n=samples, t=time)
        # -------------------

        name = f.split('/')[-1].replace('.mat', '')
        Parallel(n_jobs=parallel_jobs)(
            delayed(thread_read_write)(imgs[i], y[i], pkl_path + name+'_' + str(i) + '_'+str(y[i]))
            for i in tqdm(range(len(y)), desc=' write '+name, position=1, leave=False, colour='WHITE', ncols=80))


def file_scanf(path, endswith, sub_ratio=1):
    files = glob.glob(path + '/*')
    if platform.system().lower() == 'windows':
        files = [f.replace('\\', '/') for f in files]
    disallowed_file_endings = (".gitignore", ".DS_Store")
    _input_files = files[:int(len(files) * sub_ratio)]
    return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(endswith), _input_files))


if __name__ == "__main__":
    path = 'E:/Datasets/Stanford_digital_repository'
    filenames = file_scanf(path, endswith='.mat')
    locs = read_locs_mat('E:/Datasets/Stanford_digital_repository/electrodes_locations/Neuroscan_locs_orig.mat')
    go_through(filenames, pkl_path=path+'/img_pkl/')
