# -*- coding: UTF-8 -*-
import numpy as np
import cPickle as pickle
import scipy.io
import h5py
from load_data import loading_data
from load_data import split_data


# environmental setting: setting the following parameters based on your experimental environment.
# select_gpu = '0,1'
select_gpu = '1'
per_process_gpu_memory_fraction = 0.7

# Initialize data loader
MODEL_DIR = 'data/weight/imagenet-vgg-f.mat' #discriminator_img  pretrain model
DATA_DIR = 'data/FLICKR-25K.mat'

meanpix = np.array([0.485, 0.456, 0.406]).reshape((1,1,1,3))
meanpix = np.repeat(meanpix, 224, axis=1)
meanpix = np.repeat(meanpix, 224, axis=2)


phase = 'train'
checkpoint_dir = 'checkpoint'
Savecode = 'Savecode'
dataset_dir = 'FLICKR'
netStr = 'vgg-f'


SEMANTIC_EMBED = 512
MAX_ITER = 100
batch_size = 128
num_proposal = 8
image_size = 224


images, tags, labels = loading_data(DATA_DIR)
dimTxt = tags.shape[1]
dimLab = labels.shape[1]

DATABASE_SIZE = 4500        # 原数据设置为18015
TRAINING_SIZE = 3000        # 原数据设置为10000
QUERY_SIZE = 500            # 原数据设置为2000
VERIFICATION_SIZE = 1000
N = dimLab

X, Y, L = split_data(images, tags, labels, QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE)


train_L = L['train'].astype(np.float32)
train_x = X['train'].astype(np.float32)
train_y = Y['train'].astype(np.float32)

query_L = L['query'].astype(np.float32)
query_x = X['query'].astype(np.float32)
query_y = Y['query'].astype(np.float32)


retrieval_L = L['retrieval'].astype(np.float32)
retrieval_x = X['retrieval'].astype(np.float32)
retrieval_y = Y['retrieval'].astype(np.float32)


data = scipy.io.loadmat(MODEL_DIR)
imgMean = data['normalization'][0][0][0].astype(np.float32)
bit = 64
alpha = 1
gamma = 1
beta = 1
eta = 10
delta = 1

save_freq = 1
Epoch = 250       # 原数据设置为2000


num_train = train_x.shape[0]
numClass = train_L.shape[1]
dimText = train_y.shape[1]


Sim = (np.dot(train_L, train_L.transpose()) > 0).astype(int)*0.999

lr_img = 0.0001 #0.0001
lr_txt = 0.01 #0.001
lr_lab = 0.01
lr_gph = 0.001
# learn_rate = 0.0001
decay = 0.5
decay_steps = 1

# coffee 是个大帅哥！！！