import subprocess
import numpy as np
from cnn_bounds_full import run as run_cnn_full
from cnn_bounds_full_core import run as run_cnn_full_core
from Attacks.cw_attack import cw_attack
from CLEVER.collect_gradients import collect_gradients
from tensorflow.contrib.keras.api.keras import backend as K

import time as timing
import datetime

#时间戳
ts = timing.time()
timestr = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

#打印log文件
def printlog(s):
    print(s, file=open("log_pymain_"+timestr+".txt", "a"))

#CNN-Cert
def run_cnn(file_name, n_samples, norm, core=True, activation='relu'):
    if core:
        if norm == 'i':
            return run_cnn_full_core(file_name, n_samples, 105, 1, activation)
        elif norm == '2':
            return run_cnn_full_core(file_name, n_samples, 2, 2, activation)
        if norm == '1':
            return run_cnn_full_core(file_name, n_samples, 1, 105, activation)
    else:
        if norm == 'i':
            return run_cnn_full(file_name, n_samples, 105, 1, activation)
        elif norm == '2':
            return run_cnn_full(file_name, n_samples, 2, 2, activation)
        if norm == '1':
            return run_cnn_full(file_name, n_samples, 1, 105, activation)

#通用激活函数 CNN-Cert
def run_all_general(file_name, num_image = 10, core=True):
    if len(file_name.split('_')) == 5:
        nlayer = file_name.split('_')[-3][0]
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None

    LBs = []
    times = []
    # 测试三种范数
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []

        # 非适应ReLU
        LB, time = run_cnn(file_name, num_image, norm, core=core, activation = 'relu')
        printlog("CNN-Cert-relu")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)

        # 自适应ReLU
        LB, time = run_cnn(file_name, num_image, norm, core=core, activation = 'ada')
        printlog("CNN-Cert-Ada, ReLU activation")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)

        # Tanh
        LB, time = run_cnn(file_name + '_tanh', num_image, norm, core=core, activation = 'tanh')
        printlog("CNN-Cert-Ada, Tanh activation")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)

        # Sigmoid
        LB, time = run_cnn(file_name + '_sigmoid', num_image, norm, core=core, activation = 'sigmoid')
        printlog("CNN-Cert-Ada, Sigmoid activation")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)

        # Atan
        LB, time = run_cnn(file_name + '_atan', num_image, norm, core=core, activation = 'arctan')
        printlog("CNN-Cert-Ada, Arctan activation")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times


# Fast-Lin
def run_Fast_Lin(layers, file_name, mlp_file_name, num_image=10):
    if len(file_name.split('_')) == 5:
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        cmd = 'python3 Fast-Lin/main.py --hidden ' + str(999) + ' --numlayer ' + str(len(layers) + 1) + ' --numimage ' + str(num_image) + ' --norm ' + str(norm) + ' '
        if mlp_file_name:
            cmd += '--filename ' + str(mlp_file_name) + ' '
            cmd += '--layers ' + ' '.join(str(l) for l in layers) + ' '
        cmd += '--eps 0.05 --warmup --targettype random'

        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
        result = result.rsplit('\n', 2)[-2].split(',')
        LB = float(result[1].strip()[20:])
        time = float(result[3].strip()[17:])

        printlog("Fast-Lin")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name, len(layers) + 1, num_image, norm, filters, kernel_size))
        else:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random".format(file_name, len(layers) + 1, num_image, norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time) + " sec")
        printlog("-----------------------------------")
        LBs.append(LB)
        times.append(time)
    return LBs, times


#CLEVER
def run_CLEVER(file_name, num_image = 10):
    if len(file_name.split('_')) == 5:
        nlayer = file_name.split('_')[-3][0]
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    
    dataset = 'mnist'

    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []
        LB, time = collect_gradients(dataset, file_name, norm, num_image)
        printlog("CLEVER")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times

# CW/EAD
def run_attack(file_name, sess, num_image = 10):
    if len(file_name.split('_')) == 5:
        nlayer = file_name.split('_')[-3][0]
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    UBs = []
    times = []
    for norm in ['i', '2', '1']:
        UB, time = cw_attack(file_name, norm, sess, num_image)
        printlog("CW/EAD")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
        printlog("avg robustness = {:.5f}".format(UB))
        printlog("avg run time = {:.2f}".format(time)+" sec")
        printlog("-----------------------------------")
        UBs.append([UB])
        times.append([time])
    return UBs, times


if __name__ == '__main__':
    CNN-Cert
    print('CNN-Cert')
    LB, time = run_all_general('models/mnist_resnet_2', core=False)
    print(LB)
    print(time)

    print('CNN-Cert')
    LB, time = run_all_general('models/mnist_cnn_4layer_5_3', core=True)
    print(LB)
    print(time)

    # Fast-Lin
    print('Fast-Lin')
    LB, time = run_Fast_Lin([3380, 2880, 2420], 'models/mnist_cnn_4layer_5_3', 'mnist_4layer_relu')
    print(LB)
    print(time)

    #CLEVER
    print('CLEVER')
    LB, time = run_CLEVER('models/mnist_cnn_4layer_5_3')
    print(LB)
    print(time)

    # CW/EAD
    print('CW/EAD')
    with K.get_session() as sess:
        LB, time = run_attack('models/mnist_cnn_4layer_5_3', sess)
    K.clear_session()