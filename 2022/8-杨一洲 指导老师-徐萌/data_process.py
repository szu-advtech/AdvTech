import os
import re
import numpy as np
import h5py
import scipy.io as scio
import torch
from tqdm import tqdm

def get_month_str(month):
    if month < 10:
        month = '0' + str(month)
    return str(month)


def read_mat(path, attr):
    data = h5py.File(path, 'r')
    data = data[attr]

    return np.transpose(data)


def read_feature_mat(year, month, attr):
    if attr == 'sss':
        return read_sss(year, month)
    root_path = f'/home3/yyz_/dataset/1x1/{attr}/'
    file_name = f'{attr}_' + str(year) + get_month_str(month) + '.mat'
    path = os.path.join(root_path, file_name)
    return read_mat(path, attr)


def read_clim(month, attr):
    if attr == 'sss':
        return read_sss_clim(month)
    clim_root_path = f'/home3/yyz_/dataset/1x1/{attr}/Clim_{attr}'
    clim_file_name = f'Clim_{attr}_Mon{get_month_str(month)}.mat'
    path = os.path.join(clim_root_path, clim_file_name)
    return read_mat(path, f'Clim_{attr}')


def read_feature_minus_clim(year, month, attr):
    f = read_feature_mat(year, month, attr)
    clim = read_clim(month, attr)
    return f - clim


def read_sss(year, month):
    attr = 'sss'
    root_path = f'/home3/yyz_/dataset/1x1/{attr}/'
    file_name = f'{attr}_' + str(year) + get_month_str(month) + '.mat'
    path = os.path.join(root_path, file_name)
    data = scio.loadmat(path)
    return np.array(data[attr])


def read_sss_clim(month):
    attr = 'sss'
    clim_root_path = f'/home3/yyz_/dataset/1x1/{attr}/Clim_{attr}'
    clim_file_name = f'Clim_{attr}_Mon{get_month_str(month)}.mat'
    path = os.path.join(clim_root_path, clim_file_name)
    data = scio.loadmat(path)
    return np.array(data[f'Clim_{attr}'])


def get_feature_integrate(year, month):
    swh = read_feature_minus_clim(year, month, 'swh')
    swh = np.expand_dims(swh, 2)
    sst = read_feature_minus_clim(year, month, 'sst')
    sst = np.expand_dims(sst, 2)
    sss = read_feature_minus_clim(year, month, 'sss')
    sss = np.expand_dims(sss, 2)
    ccmp = read_feature_minus_clim(year, month, 'ccmp')
    adt = read_feature_minus_clim(year, month, 'adt')
    adt = np.expand_dims(adt, 2)
    f = swh
    f = np.concatenate((f, sst), axis=2)
    f = np.concatenate((f, sss), axis=2)
    f = np.concatenate((f, ccmp), axis=2)
    f = np.concatenate((f, adt), axis=2)
    return f


def get_feature_integrate_time_step(year, month, time_step):
    """return [10, 360, 180, 7]"""
    if month - time_step < 0:
        # 算上自己的月份， 所以+13
        start_month = month + 13 - time_step
        start_year = year - 1
    else:
        start_year = year
        start_month = month - time_step + 1

    t_year = start_year
    t_month = start_month
    f_all = np.zeros([time_step, 360, 180, 7])
    for i in range(time_step):
        print(t_year, t_month)
        f = get_feature_integrate(t_year, t_month)
        f_all[i] = f
        t_month += 1
        if t_month == 13:
            t_month = 1
            t_year = t_year + 1

    return f_all


def save_dataset_integrate_time_step_on_lon_lat(year, month, time_step):
    f_all = get_feature_integrate_time_step(year, month, time_step)
    temp_ab = get_argo_gird_ab_y(year, month, 'temp')
    salt_ab = get_argo_gird_ab_y(year, month, 'salt')

    for lon in range(360):
        for lat in range(180):

            # f_s [10, 7]
            f_s = f_all[:, lon, lat, :]
            if np.isnan(f_s).sum():
                continue
            else:
                # 拼接lon和lat
                lon_ = np.ones([time_step, 1]) * lon
                lat_ = np.ones([time_step, 1]) * lat
                f_s = np.concatenate((f_s, lon_), axis=1)
                f_s = np.concatenate((f_s, lat_), axis=1)
                # 保存温度ab 数据集 [27,]
                s_temp_ab = temp_ab[lon, lat, :]
                if not np.isnan(s_temp_ab).sum():
                    # 保存特征
                    save_root = '/home3/yyz_/dataset/1x1/dataset'
                    dir_ = 'temp/' + str(year) + '/' + str(month)
                    save_root = os.path.join(save_root, dir_)
                    os.makedirs(save_root, exist_ok=True)
                    file_name = f'{str(year)}-{str(month)}-{str(lon)}-{str(lat)}-x.ds'
                    save_path = os.path.join(save_root, file_name)
                    torch.save(f_s, save_path)
                    # 保存temp
                    temp_file_name = f'{str(year)}-{str(month)}-{str(lon)}-{str(lat)}-temp_y.ds'
                    temp_save_path = os.path.join(save_root, temp_file_name)
                    torch.save(s_temp_ab, temp_save_path)
                # 保存盐度ab 数据集[27,]
                s_salt_ab = salt_ab[lon, lat, :]
                if not np.isnan(s_salt_ab).sum():
                    # 保存特征
                    save_root = '/home3/yyz_/dataset/1x1/dataset'
                    dir_ = 'salt/' + str(year) + '/' + str(month)
                    save_root = os.path.join(save_root, dir_)
                    os.makedirs(save_root, exist_ok=True)
                    file_name = f'{str(year)}-{str(month)}-{str(lon)}-{str(lat)}-x.ds'
                    save_path = os.path.join(save_root, file_name)
                    # 保存特征
                    torch.save(f_s, save_path)
                    # 保存salt
                    temp_file_name = f'{str(year)}-{str(month)}-{str(lon)}-{str(lat)}-salt_y.ds'
                    temp_save_path = os.path.join(save_root, temp_file_name)
                    torch.save(s_temp_ab, temp_save_path)


def get_argo_gird_y(year, month, attr):
    argo_root = f'/home3/yyz_/dataset/1x1/IPRC_Argo/{attr}'
    filename = f'argo_{attr}_{year}{get_month_str(month)}.mat'
    path = os.path.join(argo_root, filename)
    return read_mat(path, attr)


def get_argo_gird_clim(month, attr):
    clim_root = '/home3/yyz_/dataset/1x1/IPRC_Argo/Clim'
    file_name = f'Argo_Clim_{attr}_Mon{get_month_str(month)}.mat'
    path = os.path.join(clim_root, file_name)
    if attr == 'salt':
        data = scio.loadmat(path)
        return np.array(data[f'clim_{attr}'])

    return read_mat(path, f"clim_{attr}")


def get_argo_gird_ab_y(year, month, attr):
    y = get_argo_gird_y(year, month, attr)
    clim = get_argo_gird_clim(month, attr)
    ab = y - clim
    return ab


if __name__ == '__main__':
    # print(get_feature_integrate(2015, 1).shape)
    # f_a = get_feature_integrate_time_step(2015, 12, 10)
    # z = f_a[0]
    #
    # v = get_feature_integrate(2015, 3)
    #
    # print(z == v)
    for year in range(2016, 2018):
        for month in tqdm(range(1, 13)):
            save_dataset_integrate_time_step_on_lon_lat(year, month, 10)
    # print(get_argo_gird_ab_y(2015, 1, 'salt').shape)