import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import *
from sklearn.model_selection import KFold

def load_data_info(source_path, target_path):
    '''
    load data info for source domain (ml-25m) and target domain (netflix)
    args:
    source_path: the path of source data
    target_path: the path of target data

    output:
    source_df: dataframe of the source data,
    target_df: dataframe of the target data,
    data_info: contains some basic info of the data, e.g. source user num,
    target user num and overlapped movie num (i.e., ns, nt, m)
    '''

    col_list = ['userId', 'movieId', 'rating', 'timestamp']
    ml_25m_df = pd.read_csv(source_path, names=col_list, encoding='utf-8', header=0)
    netflix_df = pd.read_csv(target_path, names=col_list, encoding='utf-8', header=0)

    m = len(set(ml_25m_df['movieId'].unique()) | set(netflix_df['movieId'].unique()))
    ns, ms = len(ml_25m_df['userId'].unique()), m
    nt, mt = len(netflix_df['userId'].unique()), m

    data_info = [ns, nt, m]
    source_df = ml_25m_df
    target_df = netflix_df

    return source_df, target_df, data_info

def MF(train_matrix, k, lr, lamda_U, lamda_V, max_epoch,
       domain='source', eval=True, test_matrix=None,
       batch_size=1280, verbose=True, display_step=10,
       patient=5, save_path = './MF/mf_s/mf_s.ckpt'):
    '''
    Latent Factor Model for MF implementation
    args:
    matrix: rating matrix
    k: embedding size / latent factor dimension
    lr: learning rate of the model
    lamda_U: parameter for regU regularization term
    lamda_V: parameter for regV regularization term
    epochs: iteration
    domain: source domain or target domain
    verbose: whether to print out the log info
    display_step: the interval for printing out the info
    save_path: where to save the latent factor model / embedding
    '''
    train_num = train_matrix.nnz
    train_matrix = train_matrix.toarray()
    n, m = np.shape(train_matrix)

    if eval:
        test_num = test_matrix.nnz
        test_matrix = test_matrix.toarray()

    if domain == 'target':
        save_path = './MF/mf_t/mf_t.ckpt'

    # reset the default graph
    tf.reset_default_graph()

    # ground truth
    R = tf.placeholder(tf.float32, shape=(None, m))
    mask = tf.placeholder(tf.float32, shape=(None, m))
    start = tf.placeholder(tf.int32)
    end = tf.placeholder(tf.int32)

    # user and item embedding matrix
    U = tf.get_variable('U', shape=[n, k],
                        initializer=tf.random_normal_initializer(mean=0, stddev=0.2),
                        )
    V = tf.get_variable('V', shape=[m, k],
                        initializer=tf.random_normal_initializer(mean=0, stddev=0.2))

    # For MF prediction rule: R = U^T * V / U * V^T
    pred = tf.matmul(U[start:end, :], tf.transpose(V))

    # regularization term
    regularizer_U = tf.contrib.layers.l2_regularizer(lamda_U)
    regularizer_V = tf.contrib.layers.l2_regularizer(lamda_V)
    regU = regularizer_U(U[start:end, :])
    regV = regularizer_V(tf.transpose(V))

    # loss function
    MSE = tf.reduce_sum(tf.square(tf.multiply(mask, tf.subtract(R, pred))))
    # MSE = tf.reduce_sum(tf.square(tf.multiply(mask, tf.subtract(R, pred)))) / \
    #       tf.cast(tf.count_nonzero(mask), dtype=tf.float32)
    loss = MSE + regU + regV

    # choose the GradientDescentOptimizer
    train = tf.train.AdamOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as sess:
        sess.run(init)

        epoch = 0
        stop = 0
        best_eval_MSE = np.inf
        best_epoch = 0

        #for epoch in range(epochs):
        while stop < patient and epoch <= max_epoch:
            start_idx = 0
            end_idx = start_idx + batch_size
            num_batch = np.int64(np.ceil(n / batch_size))

            total_loss = 0
            total_MSE = 0

            for i in range(num_batch):
                if end_idx > n - 1:
                    end_idx = n - 1

                batch_matrix = train_matrix[start_idx:end_idx, :]
                batch_mask = np.int64(batch_matrix > 0)
                total_MSE += sess.run(MSE,
                                      feed_dict={R: batch_matrix, mask: batch_mask, start: start_idx, end: end_idx})
                total_loss += sess.run(loss,
                                       feed_dict={R: batch_matrix, mask: batch_mask, start: start_idx, end: end_idx})
                sess.run(train, feed_dict={R: batch_matrix, mask: batch_mask, start: start_idx, end: end_idx})

                start_idx += batch_size
                end_idx += batch_size

            if (epoch + 1) % display_step == 0:
                if verbose:
                    print("Epoch: %d, Loss: %.6f, MSE: %.6f" %(epoch + 1, total_loss, total_MSE / train_num))

                if eval:
                    num_batch = np.int64(np.ceil(n / batch_size))
                    eval_loss = 0
                    eval_MSE = 0

                    for i in range(num_batch):
                        start_idx = i * batch_size
                        end_idx = (i+1) * batch_size

                        if end_idx > n - 1:
                            end_idx = n - 1

                        batch_matrix = test_matrix[start_idx:end_idx, :]
                        batch_mask = np.int64(batch_matrix > 0)
                        eval_MSE += sess.run(MSE,
                                              feed_dict={R: batch_matrix, mask: batch_mask, start: start_idx,
                                                         end: end_idx})
                        eval_loss += sess.run(loss,
                                               feed_dict={R: batch_matrix, mask: batch_mask, start: start_idx,
                                                          end: end_idx})
                    if best_eval_MSE > eval_MSE / test_num:
                        best_eval_MSE = eval_MSE / test_num
                        best_epoch = epoch + 1
                    else:
                        stop += 1
                    print("Valid-- Loss: %.6f, MSE: %.6f" % (eval_loss, eval_MSE / test_num))
                    print(stop)

            epoch += 1

        # save the MF model with important embedding info
        if not eval:
            saver = tf.train.Saver()
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            saver.save(sess, save_path=save_path)

            print('%s domain latent factor from MF model '
                  'has been well trained!' % (domain))
        else:
            return best_epoch, best_eval_MSE

        # restore test code
        # saver.restore(sess, save_path)
        # u, v = sess.run(["U:0", "V:0"])
        # print(u)
        # print(u.shape)

def df2matrix(df, n, m):
    '''
    transform one df into csr_matrix
    args:
    df: dataframe of the data
    n: user num
    m: movie num

    output:
    matrix: csr_matrix form of the data
    type <class 'scipy.sparse.csr.csr_matrix'>
    '''

    row = df['userId'].values - 1
    col = df['movieId'].values - 1
    rating = df['rating'].values

    matrix = csr_matrix((rating, (row, col)), shape=(n, m))

    return matrix

def load_matrix(source_path, target_path):
    '''
    Shortcut for calling, this function will return the matrix form of data
    '''
    source_df, target_df, data_info = load_data_info(source_path, target_path)
    ns, nt, m = data_info[0], data_info[1], data_info[2]
    source_matrix = df2matrix(source_df, ns, m)
    target_matrix = df2matrix(source_df, nt, m)

    return source_matrix, target_matrix

def KfoldValidate(df, data_info, domain='source'):
    lr = 0.001
    lamda_U = 0.05
    lamda_V = 0.05

    if domain == 'source':
        n = data_info[0]
    else:
        n = data_info[1]
    m = data_info[2]

    data_num = df.shape[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=1209)

    idx = range(data_num)
    kf_eval_MSE = []
    kf_ideal_epoch = []

    for train_idx, test_idx in kf.split(idx):
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        train_matrix = df2matrix(train_df, n, m)
        test_matrix = df2matrix(test_df, n, m)

        ideal_epoch, eval_MSE = MF(train_matrix, 100, lr=lr, lamda_U=lamda_U, lamda_V=lamda_V,
                      max_epoch=10000, test_matrix=test_matrix)

        kf_ideal_epoch.append(ideal_epoch)
        kf_eval_MSE.append(eval_MSE)

    print('For lr=%.3f, lamda_U=%.3f, lamda_V=%.3f' %(lr, lamda_U, lamda_V))
    print('Ideal Epoch: %d, Result: %.6f' %(np.mean(kf_ideal_epoch), np.mean(kf_eval_MSE)))


if __name__ == '__main__':
    # source data path
    ml_25m_rating_path = '../../data/version2/ml_25m_rating.csv'
    # target data path
    netflix_rating_path = '../../data/version2/netflix_rating.csv'

    source_df, target_df, data_info = load_data_info(ml_25m_rating_path, netflix_rating_path)

    # determine the learning rate and regularization coefficients
    # KfoldValidate(source_df, data_info, domain='source')
    KfoldValidate(target_df, data_info, domain='target')

    # pickle dump data used for quick validation
    # with open('./Mt.pkl', 'wb') as f:
    #     pickle.dump(Mt, f)
    # with open('./Ms.pkl', 'rb') as f:
    #     temp = pickle.load(f)

    # In fact, it will cost a lot of time, i have pre-executed.

    # ns = data_info[0]
    # nt = data_info[1]
    # m = data_info[2]
    #
    # source_matrix = df2matrix(source_df, ns, m)
    # target_matrix = df2matrix(target_df, nt, m)
    #
    # '''
    # args: train_matrix, k, lr, lamda_U, lamda_V, max_epoch,
    #       domain='source', eval=True, test_matrix=None,
    #       batch_size=1280, verbose=True, display_step=10,
    #       patient=5, save_path = './MF/mf_s/mf_s.ckpt'
    # '''
    #
    # MF(source_matrix, 20, 0.001, 0.01, 0.01, 40, eval=False)
    # MF(target_matrix, 20, 0.001, 0.01, 0.01, 40, 'target', eval=False)


