import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import *
from collections import defaultdict
from sklearn.model_selection import KFold
from tensorflow.contrib.layers import l2_regularizer

def df2dict(df, n):
    data_dict = defaultdict(dict)
    data_users = range(1, n+1)
    for user in data_users:
        data_dict[user] = []
        item_list = df[df['userId'] == user]['movieId'].values
        data_dict[user].extend(item_list)

    return data_dict

def diffdict(dict1, dict2):
    diff = defaultdict(dict)
    for u in dict1.keys():
        diff[u] = []
        diff_item_list = sorted(list(set(dict1[u]) - set(dict2[u])))
        diff[u].extend(diff_item_list)

    return diff

def load_datainfo(data_path, load='source'):
    '''
    load data and transform it into a dict which records the user interacted items
    args:
    data_path: input data path
    load: bool, whether to load the dict from .pkl file to accelerate the execution

    output:
    data_dict: a dict which records the user interacted items
    '''
    df = pd.read_csv(data_path, encoding='utf-8', names=['userId', 'movieId', 'rating', 'timestamp'], header=0)
    users = sorted(df['userId'].unique())
    movies = sorted(df['movieId'].unique())

    n = max(users)
    m = max(movies)

    # row = df['userId'].values - 1
    # col = df['movieId'].values - 1
    # rating = df['rating'].values
    #
    # matrix = csr_matrix((rating, (row, col)), shape=(n, m))

    if load == 'source':
        with open('./ml_25m_dict.pkl', 'rb') as f:
            data_dict = pickle.load(f)
    elif load == 'target':
        with open('./netflix_dict.pkl', 'rb') as f:
            data_dict = pickle.load(f)
    else:
        data_dict = df2dict(df, n)

    return n, m, data_dict, df

def generate_batch(data_dict, n, m, batch_size=512):
    '''
    To generate the batch data used for training and testing
    args:
    data_dict: a dict which records the user interacted items
    users: user set
    movies: movie/item set
    batch_size: the batch size of the training data

    output:
    train_batch: the batch data used for training
    '''

    users = range(1, n+1)

    batch = []
    for num in range(batch_size):
        u = random.sample(users, 1)[0]
        item_list = data_dict[u]
        if len(item_list) == 0:
            num -= 1
            continue
        i = random.sample(item_list, 1)[0]
        j = random.randint(1, m)
        while j in item_list:
            j = random.randint(1, m)
        batch.append([u, i, j])

    return np.asarray(batch)

def BPR(train_dict, test_dict, n, m, k,
        lamda_U, lamda_V, lr, max_epoch,
        domain='source', eval=True, patient=5,
        save_path='./BPR/bpr_s/bpr_s.ckpt',
        verbose=True, display_step=10):
    '''
        Latent Factor Model for BPR implementation
        args:
        train_dict: a dict which records the user interacted items for training data
        test_dict: a dict which records the user interacted items for testing data
        users: user set
        movies: movie/item set
        matrix: rating matrix
        k: embedding size / latent factor dimension
        lamda_U: parameter for regu regularization term
        lamda_V: parameter for regi and regj regularization term
        lr: learning rate of the model
        epochs: iteration
        domain: source domain or target domain
        save_path: where to save the latent factor model / embedding
        verbose: whether to print out the log info
        display_step: the interval for printing out the info
    '''
    # n = max(users)
    # m = max(movies)
    # test_num = matrix.nnz
    # R = matrix.toarray()

    if domain == 'target':
        save_path = './BPR/bpr_t/bpr_t.ckpt'

    # reset the default graph
    tf.reset_default_graph()

    # initialize the variables
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])

    U = tf.get_variable("U", shape=[n, k],
                        initializer=tf.random_normal_initializer(mean=0, stddev=0.2))
    V = tf.get_variable("V", shape=[m, k],
                        initializer=tf.random_normal_initializer(mean=0, stddev=0.2))

    u_embedding = tf.nn.embedding_lookup(U, u-1)
    i_embedding = tf.nn.embedding_lookup(V, i-1)
    j_embedding = tf.nn.embedding_lookup(V, j-1)

    # build the model
    Pr = tf.reduce_sum(tf.multiply(u_embedding, i_embedding - j_embedding), axis=1)
    auc = tf.reduce_mean(tf.to_float(Pr > 0))
    prob = -tf.reduce_mean(tf.log(tf.sigmoid(Pr)))
    regu = l2_regularizer(lamda_U)(u_embedding)
    regi = l2_regularizer(lamda_V)(i_embedding)
    regj = l2_regularizer(lamda_V)(j_embedding)

    loss = prob + regu + regi
    #loss = prob + regu + regi + regj

    train = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as sess:
        sess.run(init)

        epoch = 0
        stop = 0
        best_eval_AUC = 0
        best_epoch = 0

        #for epoch in range(epochs):
        while stop < patient and epoch <= max_epoch:
            total_loss = 0
            for num in range(m):
                uij = generate_batch(train_dict, n, m)
                batch_loss, _ = sess.run([loss, train], feed_dict={u: uij[:, 0], i:uij[:, 1], j:uij[:, 2]})
                total_loss += batch_loss

            if (epoch + 1) % display_step == 0:
                if verbose:
                    print('Avg_loss: %.6f' % (total_loss / m))

                if eval:
                    test_uij = generate_batch(test_dict, n, m)
                    test_auc = sess.run(auc, feed_dict={u: test_uij[:, 0], i: test_uij[:, 1], j: test_uij[:, 2]})

                    if best_eval_AUC <= test_auc:
                        best_eval_AUC = test_auc
                        best_epoch = epoch + 1
                    else:
                        stop += 1
                    print("Epoch: %d, AUC: %.6f" %(epoch+1, test_auc))
                    print(stop)

            epoch += 1


        # save the BPR model with important embedding info
        if not eval:
            saver = tf.train.Saver()
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            saver.save(sess, save_path=save_path)

            print('%s domain latent factor from BPR model '
                  'has been well trained!' % (domain))
        else:
            return best_epoch, best_eval_AUC

def KfoldValidate(n, m, data_dict, df):
    lr = 0.005
    lamda_U = 0.05
    lamda_V = 0.05

    data_num = df.shape[0]
    idx = range(data_num)
    kf = KFold(n_splits=5, shuffle=True, random_state=1209)

    kf_eval_AUC = []
    kf_ideal_epoch = []

    for train_idx, test_idx in kf.split(idx):
        test_df = df.loc[test_idx]

        test_dict = df2dict(test_df, n)
        train_dict = diffdict(data_dict, test_dict)

        '''
        args:train_dict, test_dict, n, m, k,
             lamda_U, lamda_V, lr, epochs,
             domain='source', save_path='./BPR/bpr_s/bpr_s.ckpt',
             verbose=True, display_step=10
        '''
        ideal_epoch, eval_AUC = BPR(train_dict, test_dict, n, m, 100, lamda_U=0.001, lamda_V=0.001, lr=0.01,
            max_epoch=10000)

        kf_ideal_epoch.append(ideal_epoch)
        kf_eval_AUC.append(eval_AUC)

    print('For lr=%.3f, lamda_U=%.3f, lamda_V=%.3f' % (lr, lamda_U, lamda_V))
    print('Ideal Epoch: %d, Result: %.6f' %(np.mean(kf_ideal_epoch), np.mean(kf_eval_AUC)))


if __name__ == '__main__':
    import random

    source_data_path = '../../data/version2/ml_25m_rating.csv'
    target_data_path = '../../data/version2/netflix_rating.csv'

    # it cost a lot of time! so I save the dict into a .pkl file...
    ns, m, source_dict, source_df = load_datainfo(source_data_path)
    nt, _, target_dict, target_df = load_datainfo(target_data_path, load='target')

    # with open('./ml_25m_dict.pkl', 'wb') as f:
    #     pickle.dump(ml_25m_dict, f)
    # with open('./netflix_dict.pkl', 'wb') as f:
    #     pickle.dump(netflix_dict, f)

    # determine the learning rate and regularization coefficients
    # KfoldValidate(ns, m, source_dict, source_df)

    # args:train_dict, test_dict, n, m, k,
    #      lamda_U, lamda_V, lr, max_epoch,
    #      domain = 'source', eval = True, patient = 5,
    #      save_path = './BPR/bpr_s/bpr_s.ckpt',
    #      verbose = True, display_step = 10

    # for source domain
    BPR(source_dict, None, ns, m, 100, 0.005, 0.005, 0.05, 50, eval=False)
    # for source domain
    BPR(target_dict, None, nt, m, 100, 0.005, 0.005, 0.05, 50, domain='target', eval=False)

