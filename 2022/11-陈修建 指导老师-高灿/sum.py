import os
import re
path = r'C:\Users\CXJ\Desktop\record_remake.txt'

only_dis_img_level_sum = 0
only_dis_pixel_level_sum = 0

l = 0
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    l = len(lines)
    for line in lines:
        a, b = re.findall(r'[0-9]+\.?[0-9]*', line)
        only_dis_pixel_level_sum += float(a)
        only_dis_img_level_sum += float(b)
print(only_dis_pixel_level_sum/l, only_dis_img_level_sum/l)
