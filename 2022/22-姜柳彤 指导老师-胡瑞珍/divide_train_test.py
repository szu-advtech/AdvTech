import shutil
import os
import csv


with open('/home/tingting/liutong/datasets-original/train_test_split.csv') as f:
    f_csv = csv.reader(f)
    # 获取headers
    headers = next(f_csv)

    for row in f_csv:
        src = os.path.join('/home/tingting/liutong/datasets-original', row[0])
        dst = os.path.join('C:')
        if row[1] == 'TRAIN':
            dst = os.path.join('/home/tingting/liutong/datasets/train')
            shutil.move(src, dst)
        elif row[1] == 'TEST':
            dst = os.path.join('/home/tingting/liutong/datasets/test')
            shutil.move(src, dst)
        else:
            print(row[1])

