import multiprocessing
import threading
import time
import sys
import os
import glob
import re

cores = int(multiprocessing.cpu_count() / 2)

batch_size_set = [500]
hiddenDim_set = [100]  # [50, 100, 150, 200]
reg_scale_set = [0.0]
optimizer_set = ['Adam']
lr_rate_set = [1e-4]  #[1e-2, 1e-3, 1e-4]

# test
dataset_set = [
    ['Rec15', 'target_train', 'auxiliary', 'target_test', 36917, 9621],
]


def exec_command(arg):
    os.system(arg)

# params coarse tuning function
def coarse_tune():
    command = []
    index = 0
    for dataset in dataset_set:
        for batch_size in batch_size_set:
            for hiddenDim in hiddenDim_set:
                for reg_scale in reg_scale_set:
                    for optimizer in optimizer_set:
                        for lr_rate in lr_rate_set:
                            cmd = 'CUDA_VISIBLE_DEVICES=0 python train.py --dataset ' + dataset[
                                0] + ' --transaction ' + dataset[1] \
                                  + ' --examination ' + dataset[2] + ' --test ' + dataset[3] \
                                  + ' --user_num ' + str(dataset[4]) + ' --item_num ' + str(dataset[5]) \
                                  + ' --batch_size ' + str(batch_size) + ' --hiddenDim ' + str(hiddenDim) \
                                  + ' --reg_scale ' + str(reg_scale) + ' --optimizer ' + str(optimizer) \
                                  + ' --lr_rate ' + str(lr_rate)
                            print(cmd)
                            command.append(cmd)

        print('\n')

    pool = multiprocessing.Pool(processes=2)
    for cmd in command:
        pool.apply_async(exec_command, (cmd,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    coarse_tune()
