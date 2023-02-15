import random
from cmath import e

from keras.datasets import cifar10, cifar100
from tensorflow.python.keras.datasets import mnist
import numpy as np

GPU_MEMORY_FRACTION_P = 0.28
DATA_TYPE = 'MNIST'
SVD_INDEX_DEEP_1 = -4  # 这里无论是深的还是浅的模型都只有两个DENSE层
SVD_INDEX_DEEP_2 = -2
MIN_NUM_WORKERS_P = 6  # 最小要参与的客户机的数量
MAX_NUM_ROUNDS_P = 400  # 最大的epoch数
DATA_SPLIT_P = (0.7, 0.15, 0.15)  # 数据分割的方式
EPOCH_PER_ROUND_P = 15  # 跑多少个batch_size
BATCH_SIZE_P = 40
NUM_CLIENTS_CONTACTED_PER_ROUND_P = 6
ROUNDS_BETWEEN_FULL_COMMUNICATION_P = [5, [11, 12, 13, 14, 0]]  # 特殊轮次需要的操作
ROUNDS_BETWEEN_VALIDATIONS_P = ROUNDS_BETWEEN_FULL_COMMUNICATION_P[0]
POWER_A_P = e
SET_VALUE = 0


def gen_non_iid_weights():
    global SET_VALUE
    if DATA_TYPE == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif DATA_TYPE == "CIFAR":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x = np.concatenate([x_train,x_test]).astype('float32')
    y = np.concatenate([y_train,y_test])

    classes_num = np.unique(y)
    classes = np.array(range(classes_num.size))
    classes_client_num = int(classes_num.shape[0] / 4)
    num_classes_this_client = random.randint(classes_client_num, classes_client_num+1)
    classes_this_client = random.sample(classes.tolist(), num_classes_this_client)

    min_train = 4000
    mid_train = 8000
    SET_VALUE = mid_train
    max_train = 12000

    weights_aa = []
    for i in range(MIN_NUM_WORKERS_P):
        w = np.array([random.random() for _ in range(num_classes_this_client)])
        weights = np.array([0.] * classes.shape[0])
        for j in range(len(classes_this_client)):
            weights[classes_this_client[j]] = w[j]
        weights /= np.sum(weights)
        if i < MIN_NUM_WORKERS_P/2:
            train_size = random.randint(min_train, mid_train)
            # print(i, "          ", train_size)
        else:
            train_size = random.randint(mid_train, max_train)

        weights_aa.append({'train_size':train_size,'class_distr':weights.tolist()})
    return weights_aa

VALUE_DATA_CLASS_TRAIN_SIZE_P = gen_non_iid_weights()

# print(VALUE_DATA_CLASS_TRAIN_SIZE_P)

large_class_dis = []
small_class_dis = []
for i in range(len(VALUE_DATA_CLASS_TRAIN_SIZE_P)):
    if VALUE_DATA_CLASS_TRAIN_SIZE_P[i]['train_size'] >= SET_VALUE:
        large_class_dis.append(VALUE_DATA_CLASS_TRAIN_SIZE_P[i]['class_distr'])
    if VALUE_DATA_CLASS_TRAIN_SIZE_P[i]['train_size'] < SET_VALUE:
        small_class_dis.append(VALUE_DATA_CLASS_TRAIN_SIZE_P[i]['class_distr'])
if DATA_TYPE == 'CIFAR_100':
    large = np.zeros(100,)
    small = np.zeros(100,)
else:
    large = np.zeros(10, )
    small = np.zeros(10, )
for i in range(len(large_class_dis)):
    large += np.array(large_class_dis[i])
for i in range(len(small_class_dis)):
    small += np.array(small_class_dis[i])
www = large / len(large_class_dis)
eee = small/ len(small_class_dis)

large_test_distr = www.tolist()
small_test_distr = eee.tolist()

all_distr=[]
for i in range(len(VALUE_DATA_CLASS_TRAIN_SIZE_P)):
    all_distr.append(VALUE_DATA_CLASS_TRAIN_SIZE_P[i]['class_distr'])
if DATA_TYPE=='CIFAR_100':
    large = np.zeros(100,)
else:
    large = np.zeros(10, )
for i in range(len(all_distr)):
    large += np.array(all_distr[i])
www = large / len(all_distr)
all_distr_ = www.tolist()
