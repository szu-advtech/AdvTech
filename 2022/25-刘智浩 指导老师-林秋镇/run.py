import random
from parameter import Electricity, ETTm2, Exchange, Illness, Weather, Traffic
import numpy as np
import torch
from exp import Exp_main
from utils.tools import dotdict

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.cuda.set_device(0)

    # default
    args = dotdict()
    args.target = 'OT'
    args.freq = 'h'
    args.checkpoints = './checkpoints/'
    args.dropout = 0.05
    args.num_workers = 0
    args.gpu = 0
    args.lradj = 'type1'
    args.devices = '0'
    args.n_heads = 8
    args.d_model = 512
    args.d_ff = 2048
    args.moving_avg = 25
    args.factor = 1
    args.patience = 3
    args.learning_rate = 0.0001
    args.batch_size = 32
    args.activation = 'gelu'
    args.loss = 'mse'
    args.train_epochs = 10

    data_dict = {
        'Electricity': Electricity,
        'ETTm2': ETTm2,
        'Exchange': Exchange,
        'Illness': Illness,
        'Weather': Weather,
        'Traffic': Traffic,
    }

    for data_type in data_dict.values():
        if data_type != Illness:
            pred_len_list = (96, 192, 336, 720)
            args.seq_len = 96  # 输入序列长度
            args.label_len = 48
        else:
            pred_len_list = (24, 36, 48, 60)
            args.seq_len = 36  # 输入序列长度
            args.label_len = 18

        for pred_len in pred_len_list:
            args.pred_len = pred_len  # 输出预测序列长度

            data_type(args)
            print('Args in experiment:')
            print(args)
            Exp = Exp_main
            for ii in range(args.itr):
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}'.format(
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor)
                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)  # setting用来保存模型的名字
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

                torch.cuda.empty_cache()

    return


if __name__ == '__main__':
    main()