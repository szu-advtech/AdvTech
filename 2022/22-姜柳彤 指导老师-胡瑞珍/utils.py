import torch
import os
import json
import dataset
import random
import time

correspond_num = 91
random_num = 2


# get all source_img's path and corresponding mask's paths
def get_path(type):
    # get the path of train_path or test_path according to parameter 'type'
    with open('config.json') as file:
        data = json.load(file)
    if type == 'train':
        dir_path = data['train_path']
    elif type == 'test':
        dir_path = data['test_path']
    else:
        raise Exception("Invalid file path!")
    # read the file under the dir_path => such as 'B000S6N026' and so on
    file_dir = os.listdir(dir_path)

    source_img = []
    source_mask = []

    label_base_color = []
    label_metallic_roughness = []
    label_normal = []

    # read the sub_dir and file
    for sub_dir in file_dir:
        # The meaning of the parameters in get_group_data:
        # dir_path (train path or test path assigned by user)
        # sub_dir (the path under the train path or test path, such as 'B000S6N026' and so on)
        # 'render', 'mask', 'base_color', 'roughness', 'normal' correspond the folder under dir_path/sub_dir
        # render has three sub folder: 0, 1, 2
        # mask, base_color, roughness, normal are all having the image correspond to render image

        # get the path of render_img
        render_group, sub_dir_count = get_group_data(dir_path, sub_dir, 'render')
        # get the path of mask_img
        mask_group = get_group_data(dir_path, sub_dir, 'mask')
        # get the path of base_color_img
        label_base_color_group = get_group_data(dir_path, sub_dir, 'base_color')
        # get the path of metallic_roughness_img
        label_metallic_roughness_group = get_group_data(dir_path, sub_dir, 'roughness')
        # get the path of normal_img
        label_normal_group = get_group_data(dir_path, sub_dir, 'normal')
        # get 40 unrepeated number, we want to get 40 different path from render_group
        # sub_dir_count * 91: all sub_dir having correspond_num images
        rand_num = random.sample(range(0, sub_dir_count * correspond_num), random_num)
        # read 40 random number
        for i in rand_num:
            # append the source image into source_img
            source_img.append(render_group[i])
            # append the correspond mask into source_mask
            source_mask.append(mask_group[i % correspond_num])
            # the same as the source_mask, we append the correspond labels into correspond list
            label_base_color.append(label_base_color_group[i % correspond_num])
            label_metallic_roughness.append(label_metallic_roughness_group[i % correspond_num])
            label_normal.append(label_normal_group[i % correspond_num])

    # return all the list
    return source_img, source_mask, label_base_color, label_metallic_roughness, label_normal


# get the file under render/mask dir
def get_group_data(dir_path, file_dir, dir_type):
    # Note: sort is very important to os.list()!!!!!!
    # The code of dir_type == 'mask', 'base_color', 'metallic', 'roughness', 'normal' /n
    # are similar to dir_type == 'render'

    # if we want to get the render image
    if dir_type == 'render':
        # get the render_img's path
        render_path = os.path.join(file_dir, 'render')
        # concat the total path, such as 'train_path / B000S6N026'
        total_path = os.path.join(dir_path, render_path)
        # get the sub_dir under the total_path
        total_path = os.listdir(total_path)
        total_path.sort(key=lambda x:int(x[0]))
        # create empty list: file_sub_list
        file_sub_list = []
        # record the number of the sub_dir
        sub_dir_count = 0
        # traversal total_path, get the sub_dir under total_path
        for rd in total_path:
            sub_dir_count += 1
            # sub_dir -> such as 'train_path / B000S6N026 / rd(0 or 1 or 2)'
            sub_dir = os.listdir(os.path.join(os.path.join(dir_path, render_path), rd))
            # sort the sub_dir according to the number of image
            sub_dir.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))
            # traversal the sub_dir
            for rd_file_name in sub_dir:
                # append the files to file_sub_list, we already sort the files below this For loop
                file_sub_list.append(os.path.join(os.path.join(render_path, rd), rd_file_name))
        return file_sub_list, sub_dir_count

    elif dir_type == 'mask':
        file_path = os.path.join(file_dir, 'segmentation')
        total_path = os.path.join(dir_path, file_path)
        mask_list = []
        for mask_file_name in os.listdir(total_path):
            mask_list.append(os.path.join(file_path, mask_file_name))
        mask_list.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))
        return mask_list

    elif dir_type == 'base_color':
        file_path = os.path.join(file_dir, 'base_color')
        total_path = os.path.join(dir_path, file_path)
        base_color_list = []
        for base_color_name in os.listdir(total_path):
            base_color_list.append(os.path.join(file_path, base_color_name))
        base_color_list.sort(key=lambda x:int(x.split('_')[3].split('.')[0]))

        return base_color_list

    elif dir_type == 'metallic' or dir_type == 'roughness':
        file_path = os.path.join(file_dir, 'metallic_roughness')
        total_path = os.path.join(dir_path, file_path)
        metallic_roughness_list = []
        for metallic_name in os.listdir(total_path):
            metallic_roughness_list.append(os.path.join(file_path, metallic_name))
        metallic_roughness_list.sort(key=lambda x:int(x.split('_')[3].split('.')[0]))

        return metallic_roughness_list

    elif dir_type == 'normal':
        file_path = os.path.join(file_dir, 'normal')
        total_path = os.path.join(dir_path, file_path)
        normal_list = []
        for normal_name in os.listdir(total_path):
            normal_list.append(os.path.join(file_path, normal_name))
        normal_list.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))

        return normal_list
