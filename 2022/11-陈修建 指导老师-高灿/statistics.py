import os
import re
path = r'/data/cxj/results/Dis3/record_dis.txt'

only_dis_img_level_sum = 0
only_dis_pixel_level_sum = 0
all_mul_img_level_sum = 0
all_mul_pixel_level_sum = 0
l = 0
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    l = len(lines)
    for line in lines:
        a, b, c, d = re.findall(r'[0-9]+\.?[0-9]*', line)
        only_dis_pixel_level_sum += float(a)
        only_dis_img_level_sum += float(b)
        all_mul_pixel_level_sum += float(c)
        all_mul_img_level_sum += float(d)
print(only_dis_pixel_level_sum/l, only_dis_img_level_sum/l, all_mul_pixel_level_sum/l, all_mul_img_level_sum/l)
