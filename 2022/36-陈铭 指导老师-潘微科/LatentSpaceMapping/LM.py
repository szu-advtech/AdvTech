import os
import pickle
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict


def load_embedding(meta_path, ckpt_path):
    '''
    load the embedding from the saved model .ckpt path
    args:
    meta_path: graph path
    ckpt_path: model saved folder

    output:
    U: user embedding matrix
    V.T: the transposition of item embedding matrix
    '''
    new_graph = tf.Graph()
    with tf.Session(graph = new_graph) as sess:
        loader = tf.train.import_meta_graph(meta_path)
        loader.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        U, V = sess.run(["U:0", "V:0"])

    return U.T, V.T

def generate_batch(data_dict, test_dict, n, m, batch_size=512):
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
        cs_item_list = test_dict[u]
        item_list = data_dict[u]

        if len(cs_item_list) == 0:
            num -= 1
            continue
        i = random.sample(cs_item_list, 1)[0]
        j = random.randint(1, m)
        while j in item_list:
            j = random.randint(1, m)
        batch.append([u, i, j])

    return np.asarray(batch)

def diffdict(dict1, sample_idx, m):
    test_dict = defaultdict(dict)
    whole_item_list = np.arange(1, m+1)
    whole_idx = np.arange(m)
    cse_idx = np.setdiff1d(whole_idx, sample_idx, False)
    cse_item_list = whole_item_list[cse_idx]

    for u in dict1.keys():
        test_dict[u] = []
        item_list = dict1[u]
        for item in item_list:
            if item in cse_item_list:
                test_dict[u].append(item)

    return test_dict

def LinearMapping(input_Vs, input_Vt, alpha, lr, Ut,
                  latent_factor_model = 'MF',
                  cold_start_entities_fraction=0.2,
                  max_epoch = 10000,
                  patience_count = 5,
                  verbose=True, display_step=10, **kwargs):
    '''
    Linear Mapping described in [Mikolov et al., 2013]
    args:
    Vs: item embedding in the source domain
    Vt: item embedding in the target domain
    alpha: the coefficient of regularization term
    lr: learning rate
    epochs: the iteration number of training
    rating_matrix: rating matrix in the target domain, used for validation
    Ut: user embedding in the target domain. used for validation
    latent_factor_model: the embedding got from the latent factor model, e.g.'MF', 'BPR'
    cold_start_entities_fraction: the fraction of the cold start entities
    max_epoch: used for early stopping
    patience_count: used for early stopping
    verbose: whether to print out the log info
    display_step: the interval for printing out the info

    output: sample_idx: mainly used for evaluation
    '''
    k, m = np.shape(input_Vs)

    # taking different fractions for cold-start entities, e.g.10%, 20%,...
    non_cold_start_entities_num = np.ceil((1-cold_start_entities_fraction) * m)
    m_sample = int(non_cold_start_entities_num)
    sample_idx = np.random.choice(range(int(m)), m_sample, replace=False)
    input_tf_Vs = input_Vs[:, sample_idx]
    input_tf_Vt = input_Vt[:, sample_idx]

    # determine the save path according to the Latent Factor Model
    save_path = './'
    if latent_factor_model == 'MF':
        save_path = './model/MF/LM'
    elif latent_factor_model == 'BPR':
        save_path = './model/BPR/LM'
        data_dict = kwargs['data_dict']
        test_dict = diffdict(data_dict, sample_idx, m)


    # if the save path doesn't exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # setting the variables
    Vs = tf.placeholder(dtype=tf.float32, shape=[k, m_sample])
    Vt = tf.placeholder(dtype=tf.float32, shape=[k, m_sample])
    M = tf.get_variable(name='M', shape=[k, k],
                        initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
    b = tf.Variable(initial_value=tf.zeros([k]), name="b")

    # build the model
    pred = tf.matmul(M, Vs) + b[:, np.newaxis]
    regM = tf.contrib.layers.l2_regularizer(alpha)(M)
    loss = tf.reduce_mean(tf.square(Vt - pred)) + regM

    if latent_factor_model == 'MF':
        train = tf.train.AdagradOptimizer(lr).minimize(loss)
    else:
        train = tf.train.AdagradOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()

    # training
    with tf.Session() as sess:
        sess.run(init)

        #for epoch in range(epochs):
        epoch = 0
        count = 0
        stop = False

        best_rmse = np.inf
        best_auc = 0
        best_epoch = 0

        while epoch < max_epoch and not stop:
            sess.run(train, feed_dict={Vs: input_tf_Vs, Vt: input_tf_Vt})

            if (epoch + 1) % display_step == 0:
                if verbose:
                    total_loss = sess.run(loss, feed_dict={Vs: input_tf_Vs, Vt: input_tf_Vt})
                    print("Epoch: %d, Loss: %.6f" %(epoch + 1, total_loss))
                    temp_M, temp_b = sess.run(["M:0", "b:0"])

                    if latent_factor_model == 'MF':
                        temp_rmse = evaluate(input_Vs, temp_M, temp_b, Ut, sample_idx, rm=kwargs['rating_matrix'])
                        if temp_rmse < best_rmse:
                            best_rmse = temp_rmse
                            best_epoch = epoch + 1
                        else:
                            count += 1
                    elif latent_factor_model == 'BPR':
                        temp_auc = evaluate(input_Vs, temp_M, temp_b, Ut, sample_idx, LFM='BPR',
                                            m=m, data_dict=data_dict, test_dict=test_dict)

                        if temp_auc > best_auc:
                            best_auc = temp_auc
                            best_epoch = epoch + 1
                            count -= 1
                        else:
                            count += 1

                    if count >= patience_count:
                        stop = True
            epoch += 1

        # print out the info of variables
        # variable_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variable_names)
        # for name, value in zip(variable_names, values):
        #     print("Variable:", name)
        #     print("Shape:", value.shape)
        #     print("Value", value)

        # print out the best performance
        print('For k=%d, csf=%.2f' % (k, cold_start_entities_fraction))
        print('lr=%.4f, alpha=%.4f' % (lr, alpha))
        if latent_factor_model == 'MF':
            print('Ideal Epoch: %d, Result: %.6f' % (best_epoch, best_rmse))
        elif latent_factor_model == 'BPR':
            print('Ideal Epoch: %d, Result: %.6f' % (best_epoch, best_auc))

        # save the model
        # saver = tf.train.Saver()
        # saver.save(sess, save_path=save_path)
        print("Linear Mapping procedure is done!")

def evaluate(Vs, M, b, Ut, sample_idx, LFM='MF', **kwargs):
    if LFM == 'MF':
        rating_matrix = kwargs['rm']

        test_num = 0
        R = rating_matrix.toarray()
        nt, mt = np.shape(R)

        whole_idx = np.arange(mt)
        cse_idx = np.setdiff1d(whole_idx, sample_idx, False)

        pred_Vt = np.dot(M, Vs[:, cse_idx]) + b[:, np.newaxis]
        U = Ut.T
        R = R[:, cse_idx]

        batch_size = 1000
        batch_num = np.int32(np.ceil(nt / batch_size))
        loss = 0

        for i in range(batch_num):
            start = i * batch_size
            end = (i + 1) * batch_size
            if end > nt:
                end = nt

            batch_R = R[start:end, :]
            mask = np.int64(batch_R > 0)
            test_num += np.sum(mask)
            pred = np.dot(U[start:end, :], pred_Vt)
            err = mask * (batch_R - pred)

            loss += np.sum(np.square(err))

        rmse = np.sqrt(loss / test_num)
        print('RMSE: %.6f' % (rmse))

        return rmse

    elif LFM == 'BPR':
        m = kwargs['m']
        _, nt = np.shape(Ut)

        data_dict = kwargs['data_dict']
        test_dict = kwargs['test_dict']
        uij = generate_batch(data_dict, test_dict, nt, m)

        u, i, j = uij[:, 0]-1, uij[:, 1]-1, uij[:, 2]-1
        U = Ut.T
        Vt = (np.dot(M, Vs) + b[:, np.newaxis]).T

        u_embed = U[u, :]
        i_embed = Vt[i, :]
        j_embed = Vt[j, :]

        pr = np.sum(u_embed * (i_embed - j_embed), axis=1)
        auc = np.mean(1 * (pr > 0))

        print('AUC: %.6f' %(auc))

        return auc

if __name__ == '__main__':
    source_model_path = '../LatentFactorModeling/MF/mf_s'
    source_meta_path = '../LatentFactorModeling/MF/mf_s/mf_s.ckpt.meta'

    target_model_path = '../LatentFactorModeling/MF/mf_t'
    target_meta_path = '../LatentFactorModeling/MF/mf_t/mf_t.ckpt.meta'

    # Us∈(100, 98907), Vs∈(100, 4100)
    Us, Vs = load_embedding(source_meta_path, source_model_path)
    # Ut∈(100, 99683), Vt∈(100, 4100)
    Ut, Vt = load_embedding(target_meta_path, target_model_path)

    with open('./Mt.pkl', 'rb') as f:
        Mt = pickle.load(f)
    LinearMapping(Vs, Vt, 0.05, 0.01, Mt, Ut)

    # sample_idx = LinearMapping(Vs, Vt, 0.002, 0.01, 200, Mt, Ut)

    # a little validation for adding the cold-start entities together
    # loading the model parameters

    # k, n = np.shape(Vs)

    # new_graph = tf.Graph()
    # meta_path = './model/MF/LM.meta'
    # ckpt_path = './model/MF'
    # with tf.Session(graph=new_graph) as sess:
    #     loader = tf.train.import_meta_graph(meta_path)
    #     loader.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    #     M, b = sess.run(["M:0", "b:0"])
    #
    # b_extend = np.zeros(n)
    # for idx in range(len(b)):
    #     b_extend[sample_idx[idx]] = b[idx]
    # pred = np.dot(M, Vs) + b_extend
    # total_loss = np.mean(np.square(pred - Vt))
    # print('Total Loss:', total_loss)
