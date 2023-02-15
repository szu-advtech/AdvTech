import os.path

import numpy as np
from collections import Counter
import random
import pandas as pd
from sklearn import ensemble


def cfs_eval(x, y):
    _, k = x.shape
    rho = np.corrcoef(np.concatenate((y, x), axis=-1), rowvar=False)
    cy = rho[0, 1:]
    rho_x = rho[1:, 1:]
    r_cf = np.mean(np.abs(cy))
    if k == 1:
        r_ff = 1
    else:
        sm = np.sum(np.sum(abs(np.triu(rho_x)))) - k
        r_ff = sm / (k * (k - 1) / 2)
    return (k * r_cf) / (np.sqrt(k + k * (k - 1) * r_ff))


# 启发式函数 x_fs，I选出来的，x_nfs，I_nfs待选的
def best_addition(x_fs, x_nfs, y, I, I_nfs):
    _, k = x_nfs.shape
    c = np.zeros((k,))
    for i in range(k):
        x = np.concatenate((x_fs, x_nfs[:, i, None]), axis=-1)
        c[i] = cfs_eval(x, y)
    c_best, i_best = np.max(c), np.argmax(c)
    x_fs = np.concatenate((x_fs, x_nfs[:, i_best, None]), axis=-1)
    x_nfs = np.concatenate((x_nfs[:, :i_best], x_nfs[:, i_best + 1:]), axis=-1)
    I = np.append(I, I_nfs[i_best])
    I_nfs = np.concatenate((I_nfs[:i_best], I_nfs[i_best + 1:]), axis=-1)
    return x_fs, x_nfs, c_best, I, I_nfs


# num样本数 k特征数
def cfs(x, y):
    num, k = x.shape
    n = 3
    I = np.array([])
    c_best = 0
    c = 0
    m = 0
    rnd = 0

    x_fs = np.empty((num, 0))
    x_nfs = x
    I_nfs = np.array(range(k))

    while len(I) < k and m < n:
        rnd = rnd + 1
        x_fs, x_nfs, c, I, I_nfs = best_addition(x_fs, x_nfs, y, I, I_nfs)

        if c > c_best + 0.0001:
            d = c - c_best
            c_best = c
            m = 0
        else:
            m = m + 1

    x_fs = x_fs[:, :-m]
    I = I[:-m]
    return x_fs, I

def connect_file_AVEC2014(file_dir,save_dir,theshold,task):

    train_file=os.path.join(file_dir,'Training',task+'.npy')
    dev_file=os.path.join(file_dir,'Development',task+'.npy')
    test_file=os.path.join(file_dir,'Testing',task+'.npy')
    data_train = np.load(train_file, allow_pickle=True)
    x_train = np.zeros((len(data_train), data_train[:, 0][0].shape[0]))
    y_train = np.zeros((len(data_train), 1))
    #

    data_test = np.load(test_file, allow_pickle=True)
    x_test = np.zeros((len(data_test), data_test[:, 0][0].shape[0]))
    y_test = np.zeros((len(data_test), 1))
    for i in range(len(data_test)):
            x_test[i] = data_test[:, 0][i]
            y_test[i] = data_test[:, 1][i]



    for i in range(len(data_train)):
        x_train[i] = data_train[:, 0][i]
        y_train[i] = data_train[:, 1][i]

    data_dev = np.load(dev_file, allow_pickle=True)

    x_dev = np.zeros((len(data_dev), data_dev[:, 0][0].shape[0]))
    y_dev = np.zeros((len(data_dev), 1))
    for i in range(len(data_dev)):
        x_dev[i] = data_dev[:, 0][i]
        y_dev[i] = data_dev[:, 1][i]

    x = np.concatenate((x_train, x_dev), axis=0)
    y = np.concatenate((y_train, y_dev), axis=0)
    print("输入的特征长度：", x.shape)
    a = np.count_nonzero(x, axis=0)
    b = np.count_nonzero(x, axis=1)
    zero_i = []
    zero_j = []
    for i in range(len(a)):
        if (a[i] <= 10):
            zero_i.append(i)

    for j in range(len(b)):
        if (b[j] == 0):
            print(y)
            zero_j.append(j)

    print("非0的个数小于10的列：", len(zero_i), zero_i)
    print("全为0的行：", zero_j)
    save_index = []
    for i in range(x.shape[1]):
        if i not in zero_i:
            save_index.append(i)
    res = []
    x=x[:,save_index]
    x_fs, I = cfs(x, y)
    res_1 = np.array(I).astype(dtype=int).tolist()
    print(res_1)
    print("全部训练集挑选出来的特征长度：", len(res_1))
    res.extend(res_1)

    divide_0=np.where(y<14)[0]
    divide_1=np.where((y>=14) &(y<20))[0]
    divide_2=np.where((y>=20) &(y<29))[0]
    divide_3=np.where(y>=29)[0]
    print(len(divide_0),len(divide_1),len(divide_2),len(divide_3))

    x_0=x[divide_0]
    y_0=y[divide_0]
    x_1=x[divide_1]
    y_1=y[divide_1]
    x_2=x[divide_2]
    y_2=y[divide_2]
    x_3=x[divide_3]
    y_3=y[divide_3]

    for i in range(10):
        random.seed(i*10)
        choose_0=random.sample([i for i in range(0, len(x_0))], 30)
        choose_1 = random.sample([i for i in range(0, len(x_1))], 10)
        choose_2=random.sample([i for i in range(0, len(x_2))], 10)
        choose_3=random.sample([i for i in range(0, len(x_3))], 10)
        x_00=x_0[choose_0]
        y_00=y_0[choose_0]
        x_11=x_1[choose_1]
        y_11=y_1[choose_1]
        x_22=x_2[choose_2]
        y_22=y_2[choose_2]
        x_33=x_3[choose_3]
        y_33=y_3[choose_3]
        x_part=np.concatenate((x_00,x_11,x_22,x_33),axis=0)
        y_part=np.concatenate((y_00,y_11,y_22,y_33),axis=0)
        x_fs_part, I_part = cfs(x_part, y_part)
        res_part = np.array(I_part).astype(dtype=int).tolist()
        print(res_part)
        print("部分训练集挑选出来的特征长度：", len(res_part))
        res.extend(res_part)

    tj = {}
    for key in res:
        tj[key] = tj.get(key, 0) + 1

    res=[]
    for key,value in tj.items():
        if value>=theshold:
            res.append(key)

    print(tj)
    print("所有筛选出的特征长度：",len(tj))
    selected = len(res)
    print(selected)
    person = np.concatenate((data_train[:, 2], data_dev[:, 2]))
    print(person.shape)
    print(x.shape)

    df_train = pd.DataFrame(columns=['feature', 'person', 'label'])
    x = x[:, res]
    df_train['feature'] = x.tolist()
    df_train['person'] = person
    df_train['label'] = y

    data_test = np.load(test_file, allow_pickle=True)
    x_test = np.zeros((len(data_test), data_test[:, 0][0].shape[0]))
    y_test = np.zeros((len(data_test), 1))
    for i in range(len(data_test)):
        x_test[i] = data_test[:, 0][i]
        y_test[i] = data_test[:, 1][i]
    df_test = pd.DataFrame(columns=['feature', 'person', 'label'])
    x_test = x_test[:, save_index]
    x_test = x_test[:, res]
    df_test['feature'] = x_test.tolist()
    df_test['person'] = data_test[:, 2]
    df_test['label'] = data_test[:, 1]

    pt=task+'_cfs'
    train_save=save_dir+"/Training/"+pt+ str(selected) + '.npy'
    test_save=save_dir+"/Testing/"+pt+ str(selected) + '.npy'
    np.save(train_save, df_train)
    np.save(test_save, df_test)
    print(train_save)
    print(test_save)
    return train_save,test_save,selected

def connect_file_AVEC2019(file_dir,save_dir,theshold):

    train_file=os.path.join(file_dir,"train.npy")
    dev_file=os.path.join(file_dir,"dev.npy")
    test_file=os.path.join(file_dir,"test.npy")
    data_train = np.load(train_file, allow_pickle=True)
    x_train = np.zeros((len(data_train), data_train[:, 0][0].shape[0]))
    y_train = np.zeros((len(data_train), 1))
    #


    for i in range(len(data_train)):
        x_train[i] = data_train[:, 0][i]
        y_train[i] = data_train[:, 1][i]

    x=x_train
    y=y_train
    print("输入的特征长度：", x.shape)
    a = np.count_nonzero(x, axis=0)
    b = np.count_nonzero(x, axis=1)
    zero_i = []
    zero_j = []
    for i in range(len(a)):
        if (a[i] <= 10):
            zero_i.append(i)

    for j in range(len(b)):
        if (b[j] == 0):
            print(y)
            zero_j.append(j)

    print("非0的个数小于10的列：", len(zero_i), zero_i)
    print("全为0的行：", zero_j)
    save_index = []
    for i in range(x.shape[1]):
        if i not in zero_i:
            save_index.append(i)
    res = []
    x=x[:,save_index]
    # save_relation=relation(x)
    # x=x[:,save_relation]
    x_fs, I = cfs(x, y)
    res_1 = np.array(I).astype(dtype=int).tolist()
    print(res_1)
    print("全部训练集挑选出来的特征长度：", len(res_1))
    res.extend(res_1)

    selected = len(res)
    print(selected)
    person=data_train[:,2]
    print(person.shape)
    print(x.shape)

    df_train = pd.DataFrame(columns=['feature', 'person', 'label'])
    x = x[:, res]
    df_train['feature'] = x.tolist()
    df_train['person'] = person
    df_train['label'] = y

    data_test = np.load(test_file, allow_pickle=True)
    x_test = np.zeros((len(data_test), data_test[:, 0][0].shape[0]))
    y_test = np.zeros((len(data_test), 1))
    for i in range(len(data_test)):
        x_test[i] = data_test[:, 0][i]
        y_test[i] = data_test[:, 1][i]
    df_test = pd.DataFrame(columns=['feature', 'person', 'label'])
    x_test = x_test[:, save_index]
    x_test = x_test[:, res]
    df_test['feature'] = x_test.tolist()
    df_test['person'] = data_test[:, 2]
    df_test['label'] = data_test[:, 1]

    data_dev = np.load(dev_file, allow_pickle=True)

    x_dev = np.zeros((len(data_dev), data_dev[:, 0][0].shape[0]))
    y_dev = np.zeros((len(data_dev), 1))
    for i in range(len(data_dev)):
        x_dev[i] = data_dev[:, 0][i]
        y_dev[i] = data_dev[:, 1][i]

    df_dev = pd.DataFrame(columns=['feature', 'person', 'label'])
    x_dev = x_dev[:, save_index]
    x_dev = x_dev[:, res]
    df_dev['feature'] = x_dev.tolist()
    df_dev['person'] = data_dev[:, 2]
    df_dev['label'] = data_dev[:, 1]

    pt='cfs'
    train_save=save_dir+"train_"+pt+str(selected) + '.npy'
    test_save=save_dir+"test_"+pt+ str(selected) + '.npy'
    dev_save=save_dir+"dev_"+pt+ str(selected) + '.npy'
    np.save(train_save, df_train)
    np.save(test_save, df_test)
    np.save(dev_save,df_dev)
    print(train_save)
    print(test_save)
    print(dev_save)
    return train_save,test_save,dev_save,selected

if __name__ == "__main__":
    #train_file,test_file,feature=connect_file_AVEC2014(r'AVEC2014/combine/features_delete','AVEC2014/combine/features_delete',2,'Freeform')
    train_file,test_file,dev_file,feature=connect_file_AVEC2019('AVEC2019/combine/features_delete','AVEC2019/combine/features_delete',2)


