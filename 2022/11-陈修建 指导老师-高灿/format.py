import os
import re
path = r'C:\Users\CXJ\Desktop\experiment_data\2022-11-30-RSTPM16-取.txt'

only_dis_img_level_sum = 0
only_dis_pixel_level_sum = 0

l = 0
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    l = len(lines)

    for line in lines:
        try:
            a, b = re.findall(r'[0-9]+\.?[0-9]*', line)
            print(round(float(a), 3), round(float(b), 3))
            only_dis_pixel_level_sum += float(a)
            only_dis_img_level_sum += float(b)
        except:
            pass
print(only_dis_pixel_level_sum/l, only_dis_img_level_sum/l)