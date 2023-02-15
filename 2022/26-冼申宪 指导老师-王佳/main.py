import argparse
import time
import os
import random
import sys
import Train
import dataset
import utils
from utils import args
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
if you have any questions
you can sent an e-mail to me
akieqi@qq.com
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-show_test', action='store_true')
    parser.add_argument('-run_tb', action='store_true')
    parser.add_argument('-run_test', action='store_true')
    parser.add_argument('-epochs', type=int, default=300)
    parser.add_argument('-es', '--early_stopping', type=int, default=50)
    parser.add_argument('-valistep', '--nb_vali_step', type=int, default=500)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-seed', type=int, default=123456)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-ds', type=str, default='yc')
    parser.add_argument('-verbose', type=int, default=1)
    parser.add_argument('-msg', type=str, default='')
    # parser.add_argument('-restore_model', type = str, default = '')
    parser.add_argument('-model', type=str, default='GNN')

    parser.add_argument('-k', '--dim_k', type=int, default=64)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-l2_all', type=float, default=1e-7)

    parser.add_argument('-seq_length', type=int, default=5)
    parser.add_argument('-adj_length', type=int, default=5)
    parser.add_argument('-num_neg', type=int, default=20)

    parser.add_argument('-gnnd', type=int, default=2)

    a = parser.parse_args().__dict__
    return a


def main(**main_args):
    begin_time = time.time()

    # init args
    args.update(**main_args)
    command_line_args = parse_args()        # 读配置
    args.setdefault(**command_line_args)        # 更新配置

    args.update(run_on_yard=True)

    seed = args.seed    #随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


    # get Model, set model default args
    #Model = vars(models)[args.model]        # 根据参数构造模型实例

    #args.setdefault(**Model.args.vars())

    if args.run_test:
        args.update(epochs=2, nb_vali_step=2, max_data_line=100)

    print(args)


    # get data
    random.seed(seed)
    np.random.seed(args.seed)
    data = dataset.Data()       # 获取到标准输入数据
    min_epochs = args.nb_train / (args.batch_size * args.nb_vali_step)
    if min_epochs < 1.0:
        args.update(nb_vali_step=int(np.ceil(args.nb_train / args.batch_size)))
        print(args)
        min_epochs = args.nb_train / (args.batch_size * args.nb_vali_step)
    args.update(min_epochs=int(np.ceil(min_epochs)))
    args.setdefault()



    # run_name: time-x-Modes-ds
    time_str = utils.get_time_str()
    model_name = 'MGNN'
    run_name = f'{time_str}-{model_name}-{args.ds}'
    if args.msg:
        run_name = f'{run_name}-{args.msg}'
    if args.run_test:
        run_name = f'{run_name}-test'

    args.update(run_name=run_name)
    T = Train.Train(data, args.dim_k)        # 构造训练实体

    # 格式输出
    log_fn = f'{utils.log_dir}/{run_name}.log'
    begin_time_str = utils.get_time_str()
    print(begin_time_str, log_fn, '----- start!, pid:', os.getpid())
    args.update(pid=os.getpid())
    log = utils.Logger(fn=log_fn, verbose=args.verbose)
    args.update(log=log)
    args.log.log(f'argv: {" ".join(sys.argv)}')
    args.log.log(f'log_fn: {log_fn}')
    args.log.log(f'args: {args.prt_json()}')
    args.log.log(f'Model: {model_name}')
    args.log.log(f'begin time: {begin_time_str}')

    try:
        T.train()
    except KeyboardInterrupt as e:
        if not T.has_train:
            raise e
    test_str = T.final_test()

    args.log.log(f'\ntest: {test_str}\n', red=True)


    args.log.log(log_fn)
    dt = time.time() - begin_time
    end_time_str = utils.get_time_str()
    args.log.log(f'end time: {end_time_str}, dt: {dt / 3600:.2f}h')
    print(end_time_str, log_fn, f'##### over, time: {dt / 3600:.2f}h')


if __name__ == '__main__':
    print(os.getcwd())
    main()

