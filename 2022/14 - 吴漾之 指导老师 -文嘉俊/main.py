#!/usr/bin/env python
import argparse
import sys
import os
import shutil
import zipfile
import time

# torchlight
import torchlight
from torchlight import import_class

from processor.processor import init_seed
init_seed(0)

def save_src(target_path):  # work_dir/crossclr_3views/1_xview_frame50_channel16_epoch300_cross150
    code_root = os.getcwd()
    srczip = zipfile.ZipFile('./src.zip', 'w')
    for root, dirnames, filenames in os.walk(code_root):
            for filename in filenames:
                if filename.split('\n')[0].split('.')[-1] == 'py':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'yaml':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'ipynb':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
    srczip.close()
    save_path = os.path.join(target_path, 'src_%s.zip' % time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))
    shutil.copy('./src.zip', save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()  # 定义一个空字典
    processors['linear_evaluation'] = import_class('processor.linear_evaluation.LE_Processor')
    processors['pretrain_crossclr_3views'] = import_class('processor.pretrain_crossclr_3views.CrosSCLR_3views_Processor') # 获取processor.pretrain_crossclr_3views中CrosSCLR_3views_Processor的值
    processors['pretrain_crossclr'] = import_class('processor.pretrain_crossclr.CrosSCLR_Processor')
    processors['pretrain_skeletonclr'] = import_class('processor.pretrain_skeletonclr.SkeletonCLR_Processor')
    # endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')  # dest用来保存匹配出来的子解析器名
    for k, p in processors.items():  # k和p分别对应键和值
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()  # 如 python main.py pretrain_crossclr_3views --config config/CrosSCLR/crossclr_3views_xview.yaml

    # start
    Processor = processors[arg.processor]  # 如processors['pretrain_crossclr_3views'],返回的是值中的对象
    p = Processor(sys.argv[2:])  # 从外部输入的参数，从.py开始算起

    if p.arg.phase == 'train':
        # save src 自定义的函数
        save_src(p.arg.work_dir)
    # 总之arg应该就是yaml配置文件，为什么没有phase字段先放着
    p.start()  # 开启线程p，应该是从上往下运行
