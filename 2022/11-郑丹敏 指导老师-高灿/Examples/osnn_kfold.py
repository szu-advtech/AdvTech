import argparse
import sys
from math import sqrt

import numpy as np

from time import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 用来生成随机整数，用于k折中每次划分已知类
import random


# Our code
sys.path.append('../MasterCodeModule/')
from OpenSetRecognition_kfold import PreprocessOSR, PerformanceMeasures, OSNN, OSNN_CV, NN, TNN, OSNN_GPD

# Set global arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='15scenes', help="15scenes | aloi | auslan | caltech256 | letter | ukbench")
parser.add_argument('--lamda_r', type=float, nargs='?', default=0.5)        # NA = lamda_r * AKS + (1 - lamda_r) * AUS 中的 lamda_r
parser.add_argument('--seed', type=int, nargs='?', default=9)
# parser.add_argument('--batch_size', type=int, nargs='?', default=128)
parser.add_argument('--k_range', type=int, nargs='+', default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
parser.add_argument('--t_range', type=float, nargs='+', default=[70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90])     # OSNN_GPD的threshold_percentile范围
parser.add_argument('--alpha', type=float, nargs='?', default=0.2)
parser.add_argument('--penalty', type=float, nargs='?', default=0.01)
parser.add_argument('--DBATCH', type=int, nargs='?', default=20000)
parser.add_argument('--num_classes', type=int, nargs='?', default=3)        # Number of classes considered to be known during training.
parser.add_argument('--n_splits', type=int, nargs='?', default=10)          # k折fold的k

#parser.add_argument('--', type=, nargs='?', default=)

# 读取参数
args = parser.parse_args()
data_filename = args.dataset
lamda_r = args.lamda_r
seed = args.seed
# batch_size = args.batch_size
k_range = args.k_range
t_range = args.t_range
alpha = args.alpha
penalty = args.penalty
DBATCH = args.DBATCH
num_classes = args.num_classes
k_range = np.array(k_range)
t_range = np.array(t_range)
n_splits = args.n_splits    # k_fold的k

start_time = time()

# OSNN方法的评价指标
AKS_osnn_array = np.zeros(shape=(n_splits,))
MIS_osnn_array = np.zeros(shape=(n_splits,))
FU_osnn_array = np.zeros(shape=(n_splits,))
AUS_osnn_array = np.zeros(shape=(n_splits,))
NA_osnn_array = np.zeros(shape=(n_splits,))
acc_osnn_array = np.zeros(shape=(n_splits,))
micF1_osnn_array = np.zeros(shape=(n_splits,))
macF1_osnn_array = np.zeros(shape=(n_splits,))

# OSNN_CV方法的评价指标
AKS_osnn_cv_array = np.zeros(shape=(n_splits,))
MIS_osnn_cv_array = np.zeros(shape=(n_splits,))
FU_osnn_cv_array = np.zeros(shape=(n_splits,))
AUS_osnn_cv_array = np.zeros(shape=(n_splits,))
NA_osnn_cv_array = np.zeros(shape=(n_splits,))
acc_osnn_cv_array = np.zeros(shape=(n_splits,))
micF1_osnn_cv_array = np.zeros(shape=(n_splits,))
macF1_osnn_cv_array = np.zeros(shape=(n_splits,))

# NN方法的评价指标
AKS_nn_array = np.zeros(shape=(n_splits,))
MIS_nn_array = np.zeros(shape=(n_splits,))
FU_nn_array = np.zeros(shape=(n_splits,))
AUS_nn_array = np.zeros(shape=(n_splits,))
NA_nn_array = np.zeros(shape=(n_splits,))
acc_nn_array = np.zeros(shape=(n_splits,))
micF1_nn_array = np.zeros(shape=(n_splits,))
macF1_nn_array = np.zeros(shape=(n_splits,))

# TNN方法的评价指标
AKS_tnn_array = np.zeros(shape=(n_splits,))
MIS_tnn_array = np.zeros(shape=(n_splits,))
FU_tnn_array = np.zeros(shape=(n_splits,))
AUS_tnn_array = np.zeros(shape=(n_splits,))
NA_tnn_array = np.zeros(shape=(n_splits,))
acc_tnn_array = np.zeros(shape=(n_splits,))
micF1_tnn_array = np.zeros(shape=(n_splits,))
macF1_tnn_array = np.zeros(shape=(n_splits,))

# OSNN_GPD方法的评价指标
AKS_osnn_gpd_array = np.zeros(shape=(n_splits,))
MIS_osnn_gpd_array = np.zeros(shape=(n_splits,))
FU_osnn_gpd_array = np.zeros(shape=(n_splits,))
AUS_osnn_gpd_array = np.zeros(shape=(n_splits,))
NA_osnn_gpd_array = np.zeros(shape=(n_splits,))
acc_osnn_gpd_array = np.zeros(shape=(n_splits,))
micF1_osnn_gpd_array = np.zeros(shape=(n_splits,))
macF1_osnn_gpd_array = np.zeros(shape=(n_splits,))


# 数据集的路径
filename = '../NNDR_feature_vectors/' + data_filename + '.dat'

print("----读取数据----")
df = pd.read_csv(filename, sep="\t")
print("pd.read_csv：", df)

# 读取vector
x_vector = df.iloc[:, 1 : df.shape[1]].values
x_vector = x_vector.astype('float32')

# 读取label
y_label = df.iloc[:, 0].values
y_label = y_label.astype('int')


print("----划分训练和测试集----")
libsvm_file_index = 0   # k折，则会生成k个训练集的libsvm格式文件以及k个测试集的libsvm格式文件
skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)     # n_splits 表示 k_fold的k
# 利用k折交叉验证，将数据集拆分成训练集和测试集
for train_ind, test_ind in skf.split(x_vector, y_label):
    train_x = x_vector[train_ind]
    train_y = y_label[train_ind]
    test_x = x_vector[test_ind]
    test_y = y_label[test_ind]

    print("train_x.shape:",train_x.shape)
    # print("train_x:",train_x)
    print("test_x.shape:",test_x.shape)
    # print("test_x:",test_x)

    print("----划分已知类和未知类----")
    # Make OSR data
    print('Creating OSR problem.')
    random_seed = random.randint(1,20)
    preprocessor = PreprocessOSR(X_vector = x_vector, y_label = y_label, random_state=random_seed)
    # train_vector, train_label = preprocessor.make_osr_data(index=keep_cls, sub_size=num_classes)        # 划分已知类和未知类后的训练集

    number_of_label = np.unique(y_label).shape[0]
    # 在所有类中随机选择一个作为划分已知类的keep_cls
    random_keep_cls = random.randint(0, number_of_label - 1)
    train_vector, train_label = preprocessor.make_osr_data(index=random_keep_cls, sub_size=num_classes)
    print("relabel后不一样的train_label：", np.unique(train_label))
    test_vector = test_x
    test_label = preprocessor.relabel_classes(test_y)
    print("relabel后不一样的test_label：", np.unique(test_label))
    print(f'The following classes were selected as the known classes for training: {list(preprocessor.reference)}')

    print("Creating OSR problem后train_vector.shape:", train_vector.shape)
    print("Creating OSR problem后test_vector.shape:", test_vector.shape)

    # 为了在svm方法上进行对比实验，需要保存每次的数据划分为libsvm格式
    print("----保存训练集数据为libsvm格式----")
    # 先把划分好的训练集的标签和特征拼接起来
    write_train = np.column_stack((train_label, train_vector))
    output = open('../NNDR_feature_vectors/write_train.txt', 'w')
    for i in range(write_train.shape[0]):
        for j in range(write_train.shape[1]):
            output.write(str(write_train[i][j]))
            output.write('\t')
        output.write('\n')
    output.close()
    # 然后再把它转换成libsvm格式的
    readin = open('../NNDR_feature_vectors/write_train.txt', 'r')
    # write data file
    output_name = '../NNDR_feature_vectors/train_libsvm_kfold' + str(libsvm_file_index)
    output = open(output_name, 'w')
    try:
        the_line = readin.readline()
        while the_line:
            # delete the \n
            the_line = the_line.strip('\n')
            index = 0;
            output_line = ''
            for sub_line in the_line.split('\t'):
                # the label col
                if index == 0:
                    output_line = float(sub_line)
                    output_line = int(output_line)
                    output_line = str(output_line)
                # the features cols
                if sub_line != 'NULL' and index != 0 and sub_line != ' ' and len(sub_line) != 0:
                    the_text = ' ' + str(index) + ':' + sub_line
                    output_line = output_line + the_text
                index = index + 1
            output_line = output_line + '\n'
            output.write(output_line)
            the_line = readin.readline()
    finally:
        readin.close()

    print("----保存测试集数据为libsvm格式----")
    # 先把划分好的测试集的标签和特征拼接起来
    write_test = np.column_stack((test_label, test_vector))
    output = open('../NNDR_feature_vectors/write_test.txt', 'w')
    for i in range(write_test.shape[0]):
        for j in range(write_test.shape[1]):
            output.write(str(write_test[i][j]))
            output.write('\t')
        output.write('\n')
    output.close()
    # 然后再把它转换成libsvm格式的
    readin = open('../NNDR_feature_vectors/write_test.txt', 'r')
    # write data file
    output_name = '../NNDR_feature_vectors/test_libsvm_kfold' + str(libsvm_file_index)
    output = open(output_name, 'w')
    try:
        the_line = readin.readline()
        while the_line:
            # delete the \n
            the_line = the_line.strip('\n')
            index = 0;
            output_line = ''
            for sub_line in the_line.split('\t'):
                # the label col
                if index == 0:
                    output_line = float(sub_line)
                    output_line = int(output_line)
                    output_line = str(output_line)
                # the features cols
                if sub_line != 'NULL' and index != 0 and sub_line != ' ' and len(sub_line) != 0:
                    the_text = ' ' + str(index) + ':' + sub_line
                    output_line = output_line + the_text
                index = index + 1
            output_line = output_line + '\n'
            output.write(output_line)
            the_line = readin.readline()
    finally:
        readin.close()

    print('----数据读取结束----')

    print('----开始使用开集识别方法----')
    # OSNN
    print('Fitting OSNN.')
    osnn = OSNN(train_vector, train_label)
    osnn.fit_threshold(threshold_range=np.linspace(0.5, 1, 10), batch_size=DBATCH,
                random_state=seed, lamda_r=lamda_r, n_splits=n_splits)  # Method to estimate the threshold of the OSNN model using a validation-set approach.

    # 获取OSNN方法识别结果
    print('Starting OSNN OSR method.')
    preds_osnn = osnn.predict(test_vector, DBATCH)
    # print("OSNN预测结果：",preds_osnn)

    ####################################################################

    # OSNN_CV
    print('Using OSNN_CV.')
    osnn_cv = OSNN_CV(train_vector, train_label)

    # 获取OSNN_CV方法识别结果
    print('Starting OSNN_CV OSR method.')
    preds_osnn_cv = osnn_cv.predict(test_vector, DBATCH)
    # print("OSNN_CV预测结果：",preds_osnn_cv)

    ####################################################################

    # NN
    print('Using NN.')
    nn = NN(train_vector, train_label)

    # 获取NN方法识别结果
    print('Starting NN method.')
    preds_nn = nn.predict(test_vector, DBATCH)
    # print("NN预测结果：",preds_nn)

    ####################################################################

    # TNN
    print('Fitting TNN.')
    tnn = TNN(train_vector, train_label)
    # 得到数据集特征数
    num_of_feature = test_vector.shape[1]
    tnn.fit_threshold(threshold_range=np.linspace(0, sqrt(num_of_feature), 100), batch_size=DBATCH,
                      random_state=seed, lamda_r=lamda_r, n_splits=n_splits)

    # 获得TNN方法识别结果
    print('Starting TNN method.')
    preds_tnn = tnn.predict(test_vector, DBATCH)
    print("TNN预测结果：",preds_tnn)

    ####################################################################

    # OSNN_GPD
    print('Fitting OSNN_GPD.')
    clf = OSNN_GPD(train_vector, train_label)
    clf.fit_parameter(t_range=np.linspace(60, 90, 100),
                 alpha=alpha,
                 penalty=penalty,
                 batch_size=DBATCH,
                 random_state=seed,
                 lamda_r = lamda_r,
                 n_splits = n_splits)

    # pars_k = clf.k
    pars_g = clf.gamma
    pars_tau = clf.tau
    pars_t = clf.threshold
    pars_tq = clf.threshold_percentile
    print(f'Fitted parameters:\ngamma={pars_g:.4f}\ntau={pars_tau:.4f}\nthreshold={pars_t:.4f}')
    # print(f'Fitted parameters:\nk={pars_k:.0f}\ngamma={pars_g:.4f}\ntau={pars_tau:.4f}\nthreshold={pars_t:.4f}')
    print(f'The threshold of {pars_t:.4f} is the {pars_tq:.2f}th percentile of the distance ratio.')

    # 获取OSNN_GPD方法识别结果
    print('Starting OSNN_GPD OSR method.')

    preds_osnn_gpd, pvals_osnn_gpd = clf.predict(X_test=test_vector,
                                                 alpha=alpha,
                                                 batch_size=DBATCH)
    print("OSNN_GPD预测结果：", preds_osnn_gpd)

    ####################################################################

    # 计算评价指标
    pm = PerformanceMeasures()
    # OSNN
    AKS_osnn_array[libsvm_file_index], MIS_osnn_array[libsvm_file_index], FU_osnn_array[libsvm_file_index] = pm.AKS(test_label, preds_osnn)
    AUS_osnn_array[libsvm_file_index] = pm.AUS(test_label, preds_osnn)
    NA_osnn_array[libsvm_file_index] = pm.NA(test_label, preds_osnn, lamda_r=lamda_r)
    acc_osnn_array[libsvm_file_index] = pm.accuracy(test_label, preds_osnn)
    micF1_osnn_array[libsvm_file_index] = pm.micro_F1(test_label, preds_osnn)
    macF1_osnn_array[libsvm_file_index] = pm.macro_F1(test_label, preds_osnn)

    # OSNN_CV
    AKS_osnn_cv_array[libsvm_file_index], MIS_osnn_cv_array[libsvm_file_index], FU_osnn_cv_array[libsvm_file_index] = pm.AKS(
        test_label, preds_osnn_cv)
    AUS_osnn_cv_array[libsvm_file_index] = pm.AUS(test_label, preds_osnn_cv)
    NA_osnn_cv_array[libsvm_file_index] = pm.NA(test_label, preds_osnn_cv, lamda_r=lamda_r)
    acc_osnn_cv_array[libsvm_file_index] = pm.accuracy(test_label, preds_osnn_cv)
    micF1_osnn_cv_array[libsvm_file_index] = pm.micro_F1(test_label, preds_osnn_cv)
    macF1_osnn_cv_array[libsvm_file_index] = pm.macro_F1(test_label, preds_osnn_cv)

    # NN
    AKS_nn_array[libsvm_file_index], MIS_nn_array[libsvm_file_index], FU_nn_array[libsvm_file_index] = pm.AKS(
        test_label, preds_nn)
    AUS_nn_array[libsvm_file_index] = pm.AUS(test_label, preds_nn)
    NA_nn_array[libsvm_file_index] = pm.NA(test_label, preds_nn, lamda_r=lamda_r)
    acc_nn_array[libsvm_file_index] = pm.accuracy(test_label, preds_nn)
    micF1_nn_array[libsvm_file_index] = pm.micro_F1(test_label, preds_nn)
    macF1_nn_array[libsvm_file_index] = pm.macro_F1(test_label, preds_nn)

    # TNN
    AKS_tnn_array[libsvm_file_index], MIS_tnn_array[libsvm_file_index], FU_tnn_array[libsvm_file_index] = pm.AKS(
        test_label, preds_tnn)
    AUS_tnn_array[libsvm_file_index] = pm.AUS(test_label, preds_tnn)
    NA_tnn_array[libsvm_file_index] = pm.NA(test_label, preds_tnn, lamda_r=lamda_r)
    acc_tnn_array[libsvm_file_index] = pm.accuracy(test_label, preds_tnn)
    micF1_tnn_array[libsvm_file_index] = pm.micro_F1(test_label, preds_tnn)
    macF1_tnn_array[libsvm_file_index] = pm.macro_F1(test_label, preds_tnn)

    #OSNN_GPD
    AKS_osnn_gpd_array[libsvm_file_index], MIS_osnn_gpd_array[libsvm_file_index], FU_osnn_gpd_array[libsvm_file_index] = pm.AKS(
        test_label, preds_osnn_gpd)
    AUS_osnn_gpd_array[libsvm_file_index] = pm.AUS(test_label, preds_osnn_gpd)
    NA_osnn_gpd_array[libsvm_file_index] = pm.NA(test_label, preds_osnn_gpd, lamda_r=lamda_r)
    acc_osnn_gpd_array[libsvm_file_index] = pm.accuracy(test_label, preds_osnn_gpd)
    micF1_osnn_gpd_array[libsvm_file_index] = pm.micro_F1(test_label, preds_osnn_gpd)
    macF1_osnn_gpd_array[libsvm_file_index] = pm.macro_F1(test_label, preds_osnn_gpd)

    libsvm_file_index += 1

# 计算k折平均评价指标
print("AKS_osnn_array:",AKS_osnn_array)
# OSNN
AKS_osnn = np.mean(AKS_osnn_array, axis=0)  # 按行求平均
MIS_osnn = np.mean(MIS_osnn_array, axis=0)
FU_osnn = np.mean(FU_osnn_array, axis=0)
AUS_osnn = np.mean(AUS_osnn_array, axis=0)
NA_osnn = np.mean(NA_osnn_array, axis=0)
acc_osnn = np.mean(acc_osnn_array, axis=0)
micF1_osnn = np.mean(micF1_osnn_array, axis=0)
macF1_osnn = np.mean(macF1_osnn_array, axis=0)

# OSNN_CV
print("AKS_osnn_cv_array:",AKS_osnn_cv_array)
AKS_osnn_cv = np.mean(AKS_osnn_cv_array, axis=0)  # 按行求平均
MIS_osnn_cv = np.mean(MIS_osnn_cv_array, axis=0)
FU_osnn_cv = np.mean(FU_osnn_cv_array, axis=0)
AUS_osnn_cv = np.mean(AUS_osnn_cv_array, axis=0)
NA_osnn_cv = np.mean(NA_osnn_cv_array, axis=0)
acc_osnn_cv = np.mean(acc_osnn_cv_array, axis=0)
micF1_osnn_cv = np.mean(micF1_osnn_cv_array, axis=0)
macF1_osnn_cv = np.mean(macF1_osnn_cv_array, axis=0)

# NN
print("AKS_nn_array:",AKS_nn_array)
AKS_nn = np.mean(AKS_nn_array, axis=0)  # 按行求平均
MIS_nn = np.mean(MIS_nn_array, axis=0)
FU_nn = np.mean(FU_nn_array, axis=0)
AUS_nn = np.mean(AUS_nn_array, axis=0)
NA_nn = np.mean(NA_nn_array, axis=0)
acc_nn = np.mean(acc_nn_array, axis=0)
micF1_nn = np.mean(micF1_nn_array, axis=0)
macF1_nn = np.mean(macF1_nn_array, axis=0)

# TNN
print("AKS_tnn_array:",AKS_tnn_array)
AKS_tnn = np.mean(AKS_tnn_array, axis=0)  # 按行求平均
MIS_tnn = np.mean(MIS_tnn_array, axis=0)
FU_tnn = np.mean(FU_tnn_array, axis=0)
AUS_tnn = np.mean(AUS_tnn_array, axis=0)
NA_tnn = np.mean(NA_tnn_array, axis=0)
acc_tnn = np.mean(acc_tnn_array, axis=0)
micF1_tnn = np.mean(micF1_tnn_array, axis=0)
macF1_tnn = np.mean(macF1_tnn_array, axis=0)

# OSNN_GPD
print("AKS_osnn_gpd_array:",AKS_osnn_gpd_array)
AKS_osnn_gpd = np.mean(AKS_osnn_gpd_array, axis=0)  # 按行求平均
MIS_osnn_gpd = np.mean(MIS_osnn_gpd_array, axis=0)
FU_osnn_gpd = np.mean(FU_osnn_gpd_array, axis=0)
AUS_osnn_gpd = np.mean(AUS_osnn_gpd_array, axis=0)
NA_osnn_gpd = np.mean(NA_osnn_gpd_array, axis=0)
acc_osnn_gpd = np.mean(acc_osnn_gpd_array, axis=0)
micF1_osnn_gpd = np.mean(micF1_osnn_gpd_array, axis=0)
macF1_osnn_gpd = np.mean(macF1_osnn_gpd_array, axis=0)

print('All methods completed. Gathering results.')
methods = ['OSNN', 'OSNN_CV', 'NN', 'TNN', 'OSNN_GPD']
AKS = [AKS_osnn, AKS_osnn_cv, AKS_nn, AKS_tnn, AKS_osnn_gpd]
MIS = [MIS_osnn, MIS_osnn_cv, MIS_nn, MIS_tnn, MIS_osnn_gpd]
FU = [FU_osnn, FU_osnn_cv, FU_nn, FU_tnn, FU_osnn_gpd]
AUS = [AUS_osnn, AUS_osnn_cv, AUS_nn, AUS_tnn, AUS_osnn_gpd]
NA = [NA_osnn, NA_osnn_cv, NA_nn, NA_tnn, NA_osnn_gpd]
Accs = [acc_osnn, acc_osnn_cv, acc_nn, acc_tnn, acc_osnn_gpd]
micF1 = [micF1_osnn, micF1_osnn_cv, micF1_nn, micF1_tnn, micF1_osnn_gpd]
macF1 = [macF1_osnn, macF1_osnn_cv, macF1_nn, macF1_tnn, macF1_osnn_gpd]

result = pd.DataFrame(list(zip(AKS, MIS, FU, AUS, NA, micF1, macF1, Accs)),
                     columns=['AKS', 'MIS', 'FU', 'AUS', 'NA', 'micF1', 'macF1', 'Accuracy'],
                     index=methods)
result_show = result.T.reindex(['AKS', 'MIS', 'FU', 'AUS', 'NA', 'micF1', 'macF1', 'Accuracy'])

df_results = pd.DataFrame(data=np.zeros(shape=result_show.shape, dtype=str), columns=result_show.columns,
                          index=result_show.index)
for row in range(result_show.shape[0]):
    for col in range(result_show.shape[1]):
        df_results.iloc[row, col] = '{:.2f}'.format(result_show.iloc[row, col] * 100)
print(df_results)
