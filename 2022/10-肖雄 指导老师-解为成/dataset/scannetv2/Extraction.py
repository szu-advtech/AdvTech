import sys
import glob
import os
import shutil

"""
python download-scannet.py -o "D:\Postgraduate\qianyan\Stratified-Transformer\dataset\scannetv2\_vh_clean_2.ply" --type _vh_clean_2.ply
python download-scannet.py -o "D:\Postgraduate\qianyan\Stratified-Transformer\dataset\scannetv2\_vh_clean_2.0.010000.segs.json" --type _vh_clean_2.0.010000.segs.json
python download-scannet.py -o "D:\Postgraduate\qianyan\Stratified-Transformer\dataset\scannetv2\_vh_clean_2.labels.ply" --type _vh_clean_2.labels.ply
"""


criteria = []   # 按照哪个表格来复制

tablename = 'test.txt'
dirname = '_vh_clean_2.ply' # '_vh_clean_2.ply'  '_vh_clean_2.0.010000.segs.json'  '_vh_clean_2.labels.ply'  '.aggregation.json'
tar_dir = 'test'   # 'val' 'train' 'test'

def table_Ex(tablename):
    with open(tablename,'r+') as f:
        for line in f.readlines():
            criteria.append(line[:-1])

    return criteria


def dir_Cp(dirname):
    criteria = table_Ex(tablename)

    sets = glob.glob(dirname + '/scans_test/*', recursive=True)


    for set in sets:
        if set[-12:] in criteria:
            src = "".join(glob.glob(set + '/*', recursive=True))
            shutil.copy(src, tar_dir)


if __name__ == '__main__':
    dir_Cp(dirname)