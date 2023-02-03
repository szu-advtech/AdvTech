import numpy as np
from numpy import load
from PIL import Image
import torch

cuda = True if torch.cuda.is_available() else False
T = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def load_real_data(filename):

    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']

    # normalize from [0,255] to [-1,1]
    #X1 = (X1 - 127.5) / 127.5
    #X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

def generate_real_data(data, random_samples, patch_shape):

    trainA, trainB = data

    id = np.random.randint(0, trainA.shape[0], random_samples) #从0-850之间选择4个随机数
    #print(id)

    X1, X2 = trainA[id], trainB[id]
    #print(X1.shape)
    #print(X2.shape)

    # generate 'real' class labels (1) 不同尺度的真实图片label
    y1 = np.ones((random_samples, 1, patch_shape[0], patch_shape[0]))
    y2 = np.ones((random_samples, 1, patch_shape[1], patch_shape[1]))
    y3 = np.ones((random_samples, 1, patch_shape[2], patch_shape[2]))

    y1 = torch.from_numpy(y1).type(T)
    y2 = torch.from_numpy(y2).type(T)
    y3 = torch.from_numpy(y3).type(T)

    #print(y1.shape)
    #print(y2.shape)
    #print(y3.shape)
    return [X1, X2], [y1,y2,y3]

def resize(X_realA, X_realB, out_shape):
    #print(X_realA[1][200][200][2])
    #print(X_realB[1][200][200][0])
    # import tensorflow as tf
    # X_realA = tf.image.resize(X_realA, out_shape, method=tf.image.ResizeMethod.LANCZOS3)
    # X_realA = np.array(X_realA)
    # X_realB = tf.image.resize(X_realB, out_shape, method=tf.image.ResizeMethod.LANCZOS3)
    # X_realB = np.array(X_realB)
    # print(X_realA[1][200][200][2])
    # print(X_realB[1][200][200][0])

    xrealA = list()
    for batchsize in range(X_realA.shape[0]):
        #print(batchsize)
        X_realA_bs = Image.fromarray(X_realA[batchsize])
        X_realA_bs = X_realA_bs.resize(out_shape,resample=Image.LANCZOS)
        X_realA_bs = np.array(X_realA_bs)
        xrealA.append(X_realA_bs)
    xrealA = np.asarray(xrealA)
    #print(xrealA.shape)
    #print(type(xrealA))
    #print(xrealA[1][200][200][2])

    xrealB = list()
    for batchsize in range(X_realB.shape[0]):
        X_realB_bs = Image.fromarray(X_realB[batchsize].squeeze(axis=2))
        X_realB_bs = X_realB_bs.resize(out_shape, resample=Image.LANCZOS)
        X_realB_bs = np.array(X_realB_bs)
        xrealB.append(X_realB_bs)
    xrealB = np.asarray(xrealB)
    #print(xrealB.shape)
    #print(xrealB[1][200][200])
    #print(type(xrealB))
    xrealB = np.expand_dims(xrealB,axis=-1)
    #print(xrealB.shape)
    #print(xrealB[1][200][200][0])

    return [xrealA, xrealB]

def generate_fake_data_coarse(coarse_generator, batch_data, patch_shape):
    # generate fake coarse data
    X, X_global = coarse_generator(batch_data)

    # create 'fake' class labels (0)
    y1 = np.zeros((len(X), 1, patch_shape[1], patch_shape[1]))
    y1 = torch.from_numpy(y1).type(T)
    y2 = np.zeros((len(X), 1, patch_shape[2], patch_shape[2]))
    y2 = torch.from_numpy(y2).type(T)
    return [X,X_global], [y1,y2]

def generate_fake_data_fine(fine_generator, batch_data, X_feature, patch_shape):
    # generate fake fine data
    X = fine_generator((batch_data,X_feature))

    # create 'fake' class labels (0)
    y1 = np.zeros((len(X), 1, patch_shape[0], patch_shape[0]))
    y1 = torch.from_numpy(y1).type(T)
    y2 = np.zeros((len(X), 1, patch_shape[1], patch_shape[1]))
    y2 = torch.from_numpy(y2).type(T)
    return X, [y1,y2]