"""
生成掩码和合成数据
"""


import numpy as np
import torch


# 生成伪标记矩阵
def generate_pseudo_masks(args, num_samples):
    # xxxx
    # xxxx
    # xxxx
    masks = np.zeros((num_samples, args.ts_size), dtype=bool)
    return masks


# 批量随机生成真实标记矩阵
def generate_random_masks(args, num_samples):
    # xxxo
    # oxxx
    # xxox

    # 可以掩盖的可能数
    num_patches = int(args.ts_size // args.mask_size)

    # 随机掩盖
    def single_sample_mask():
        # 随机打乱产生一个列表，并从中随机选取num_masks个值，即要掩盖的位置
        idx = np.random.permutation(num_patches)[:args.num_masks]
        mask = np.zeros(args.ts_size, dtype=bool)
        for j in idx:
            mask[j * args.mask_size:(j + 1) * args.mask_size] = 1
        return mask

    # _ 表示临时或无意义的变量
    masks_list = [single_sample_mask() for _ in range(num_samples)]
    masks = np.stack(masks_list, axis=0)  # (num_samples, ts_size)
    return masks


def generate_cross_masks(args, num_samples, idx):
    # oxxx
    # oxxx
    # oxxx
    masks = np.zeros((num_samples, args.ts_size), dtype=bool)
    masks[:, idx * args.total_mask_size:(idx + 1) * args.total_mask_size] = 1  # masks(num_samples, ts_size)
    return masks

"""Generation for the Evaluation and Synthesis
    1. full_generation: no patches masked, Auto Encoder style
    2. random_generation: randomly generate some masks
    3. random_average_generation_engine: random mask, average results
    4. cross_concate_generation_engine: cross mask, concate patches
    5. cross_average_generation_engine: cross mask, average results"""


# 无补丁掩码，传统AE
def full_generation(args, model, ori_data):
    # 此时获得的mask全为false
    masks = generate_pseudo_masks(args, len(ori_data))  # (len(ori_data), seq_len)
    # 然后在train_mae的mask步骤中mask会全部取反，相当所有数据没有被掩盖
    x_enc, art_data, masks = model(ori_data, masks, 'full_generation')
    np.save(args.masks_dir, masks)
    return art_data


# 随即生成一些掩码，MAE
def random_generation(args, model, ori_data):
    masks = generate_random_masks(args, len(ori_data))  # (len(ori_data), seq_len)
    x_enc, art_data, masks = model(ori_data, masks, 'random_generation')
    np.save(args.masks_dir, masks)
    return art_data


# 随机生成一些掩码且将生成数据的平均值作为最终的生成数据
def random_average_generation(args, model, ori_data):
    # 取十次生成数据的平均值作为最终的生成数据
    num_gen = 10
    generations = []
    for i in range(num_gen):
        masks = generate_random_masks(args, len(ori_data))
        _, generation, masks = model(ori_data, masks, 'random_average_generation')
        generations.append(generation)
    generations = torch.stack(generations)
    art_data = torch.mean(generations, dim=0, keepdim=False)
    return art_data


# 交叉掩码，然后将生成的片段拼接起来作为最终的生成数据
def cross_concat_generation(args, model, ori_data):
    # eg. num_mask = 2 mask_size = 4 ts_size = 20
    # args.ts_size//args.mask_size = 5
    # oo|oo|o where o = 8 positions
    # num_gen = 2
    # num_rest = 1
    num_gen = int(args.ts_size//args.total_mask_size)
    split_pos = num_gen * args.total_mask_size
    generations = []
    for i in range(num_gen):
        masks = generate_cross_masks(args, len(ori_data), i)
        _, generation, masks = model(ori_data, masks, 'cross_concat_generation')
        # 只把生成的数据中之前进行掩盖的部分取出来保存
        generations.append(generation[:, i * args.total_mask_size:(i+1) * args.total_mask_size, :])
    if split_pos != args.ts_size:
        masks = np.zeros((len(ori_data), args.ts_size), dtype=bool)
        masks[:, -args.total_mask_size:] = 1
        _, generation, masks = model(ori_data, masks, 'cross_concat_generation')
        generations.append(generation[:, split_pos:, :])
    # 将各片段按列拼接起来
    art_data = torch.cat(generations, dim=1)
    return art_data


# 交叉掩码，对交叉掩码生成的数据合并后取这些数据的平均值作为交叉掩盖生成数据
def cross_average_generation(args, model, ori_data):
    # 计算可以掩盖生成的个数  序列片段长度/掩盖总数
    num_gen = int(args.ts_size//args.total_mask_size)
    split_pos = num_gen * args.total_mask_size
    generations = []
    for i in range(num_gen):
        masks = generate_cross_masks(args, len(ori_data), i)
        _, generation, masks = model(ori_data, masks, 'cross_average_generation')
        # 将掩盖每个片段后生成的数据保存到一个列表中
        generations.append(generation)
    # 此处表示样本有剩余
    if split_pos != args.ts_size:
        masks = np.zeros((len(ori_data), args.ts_size), dtype=bool)
        masks[:, -args.total_mask_size:] = 1
        _, generation, masks = model(ori_data, masks, 'cross_average_generation')
        generations.append(generation)

    # 将对各段掩盖后生成的数据合并取平均值作为交叉掩盖的生成数据
    generations = torch.stack(generations)
    art_data = torch.mean(generations, dim=0, keepdim=False)
    return art_data
