import torch
from torchvision import datasets
import os
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

qp_min = 0
qp_max = 51

H = 64
W = 64

for qp in range(qp_max, qp_min, -1):
    cls_lst = os.listdir('tiny200/val')
    ret_enc_path = f'ori_img_qp{qp}_enc/val'
    ret_dec_path = f'ori_img_qp{qp}_dec/val'
    enc_total_size = 0
    dec_total_size= 0
    dataset_size = 0
    for cls in cls_lst:
        ori_path = os.path.join('tiny200/val', cls)

        images = os.listdir(ori_path)
        images = natural_sort(images)

        os.system(f'mkdir -p ori_img_qp{qp}_enc/val/{cls}')
        os.system(f'mkdir -p ori_img_qp{qp}_dec/val/{cls}')

        for i, image in enumerate(images):
            print(f'qp:{qp}, class:{cls}, processing {i}th image: {image}')
            ori_path_name = os.path.join(ori_path, image)
            ret_enc_path_name = os.path.join(ret_enc_path, cls, image.split('.')[0]+'.bpg')
            ret_dec_path_name = os.path.join(ret_dec_path, cls, image)

            os.system(f'bpgenc -o {ret_enc_path_name} -q {qp} {ori_path_name}')
            enc_total_size += os.path.getsize(ret_enc_path_name)
            os.system(f'bpgdec -o {ret_dec_path_name} {ret_enc_path_name}')
            dec_total_size += os.path.getsize(ret_dec_path_name)
            dataset_size += 1

    enc_average_size = enc_total_size / dataset_size
    dec_average_size = dec_total_size / dataset_size
    enc_bpp = enc_average_size * 8 / H / W
    dec_bpp = dec_average_size * 8 / H / W

    file_obj = open(os.path.join(ret_enc_path, f'result.txt'), 'w')
    file_obj.write(f'enc_average_size: {enc_average_size}\nenc_bpp: {enc_bpp}\ndataset_size: {dataset_size}')
    file_obj.close()

    file_obj = open(os.path.join(ret_dec_path, f'result.txt'), 'w')
    file_obj.write(f'dec_average_size: {dec_average_size}\ndec_bpp: {dec_bpp}\ndataset_size: {dataset_size}')
    file_obj.close()
