import os
import pickle
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib.layers import l2_regularizer

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

def MultiLayerPerceptron(input_Vs, input_Vt, Ut, alpha, lr,
                         latent_factor_model = 'MF',
                         cold_start_entities_fraction=0.2,
                         max_epoch = 10000,
                         patience_count = 5,
                         verbose=True, display_step=10, **kwargs):

    k, m = np.shape(input_Vs)

    # taking different fractions for cold-start entities, e.g.10%, 20%,...
    non_cold_start_entities_num = np.ceil((1 - cold_start_entities_fraction) * m)
    m_sample = int(non_cold_start_entities_num)
    sample_idx = np.random.choice(range(int(m)), m_sample, replace=False)
    input_tf_Vs = input_Vs[:, sample_idx]
    input_tf_Vt = input_Vt[:, sample_idx]

    # determine the save path according to the Latent Factor Model
    save_path = './'
    if latent_factor_model == 'MF':
        save_path = './model/MF/MLP'
        rating_matrix = kwargs['rating_matrix']
    elif latent_factor_model == 'BPR':
        save_path = './model/BPR/MLP'
        test_dict = diffdict(kwargs['data_dict'], sample_idx, m)

    # if the save path doesn't exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # setting the variables
    Vs = tf.placeholder(dtype=tf.float32, shape=[k, m_sample])
    Vt = tf.placeholder(dtype=tf.float32, shape=[k, m_sample])

    if latent_factor_model == 'MF':

        w1 = tf.Variable(tf.random.normal([2 * k, k], stddev=0.02), name="w1")
        w2 = tf.Variable(tf.random.normal([k, 2 * k], stddev=0.02), name="w2")

    elif latent_factor_model == 'BPR':
        # initialization follow in [Glorot and Bengio, 2010].
        a1 = 1 / np.sqrt(k)
        a2 = 1 / np.sqrt(2 * k)

        w1 = tf.Variable(tf.random.normal([2 * k, k], stddev=a1), name="w1")
        w2 = tf.Variable(tf.random.normal([k, 2 * k], stddev=a2), name="w2")

    b1 = tf.Variable(tf.zeros([2 * k, 1]), name="b1")
    b2 = tf.Variable(tf.zeros([k, 1]), name="b2")

    # build the model
    # MLP as one-hidden layer, i.e., K -> 2K -> K
    # tan-sigmoid function employed as the activation function

    if latent_factor_model == 'MF':
        # hidden = tf.nn.sigmoid(tf.matmul(w1, Vs) + b1)
        # pred = tf.nn.tanh(tf.matmul(w2, hidden) + b2)

        # hidden = tf.nn.tanh(tf.matmul(w1, Vs) + b1)
        # pred = tf.nn.sigmoid(tf.matmul(w2, hidden) + b2)

        # hidden = tf.nn.relu(tf.matmul(w1, Vs) + b1)
        # pred = tf.matmul(w2, hidden) + b2

        hidden = tf.nn.tanh(tf.matmul(w1, Vs) + b1)
        pred = tf.matmul(w2, hidden) + b2
    else:
        # sigmoid-tanh
        hidden = tf.nn.sigmoid(tf.matmul(w1, Vs) + b1)
        pred = tf.nn.tanh(tf.matmul(w2, hidden) + b2)

        # tanh-sigmoid
        # hidden = tf.nn.tanh(tf.matmul(w1, Vs) + b1)
        # pred = tf.nn.sigmoid(tf.matmul(w2, hidden) + b2)

        # tanh-none
        # hidden = tf.nn.tanh(tf.matmul(w1, Vs) + b1)
        # pred = tf.matmul(w2, hidden) + b2

        # sigmoid-none
        # hidden = tf.nn.sigmoid(tf.matmul(w1, Vs) + b1)
        # pred = tf.matmul(w2, hidden) + b2

    reg_w1 = l2_regularizer(alpha)(w1)
    reg_w2 = l2_regularizer(alpha)(w2)

    loss = tf.reduce_mean(tf.square(Vt - pred)) + reg_w1 + reg_w2

    if latent_factor_model == 'MF':
        train = tf.train.AdagradOptimizer(lr).minimize(loss)
    else:
        train = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    init = tf.global_variables_initializer()

    # training
    with tf.Session() as sess:
        sess.run(init)

        # for epoch in range(epochs):
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
                    print("Epoch: %d, Loss: %.6f" % (epoch + 1, total_loss))

                    temp_w1, temp_b1, temp_w2, temp_b2 = \
                        sess.run(["w1:0", "b1:0", "w2:0", "b2:0"])
                    if latent_factor_model == 'MF':
                        temp_rmse = MF_evaluate(input_Vs, rating_matrix, Ut, sample_idx,
                                                w1=temp_w1, b1=temp_b1, w2=temp_w2, b2=temp_b2)

                        if temp_rmse < best_rmse:
                            best_rmse = temp_rmse
                            best_epoch = epoch + 1
                        else:
                            count += 1

                    elif latent_factor_model == 'BPR':
                        temp_auc = BPR_evaluate(input_Vs, Ut, sample_idx, m=m, test_dict=test_dict,
                                                w1=temp_w1, b1=temp_b1, w2=temp_w2, b2=temp_b2)

                        if temp_auc > best_auc:
                            best_auc = temp_auc
                            best_epoch = epoch + 1
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
        print('lr=%.6f, alpha=%.6f' % (lr, alpha))
        if latent_factor_model == 'MF':
            print('Ideal Epoch: %d, Result: %.6f' % (best_epoch, best_rmse))
        elif latent_factor_model == 'BPR':
            print('Ideal Epoch: %d, Result: %.6f' % (best_epoch, best_auc))

        # save the model
        # saver = tf.train.Saver()
        # saver.save(sess, save_path=save_path)
        print("Non-Linear Mapping procedure is done!")

def MF_evaluate(Vs, rating_matrix, Ut, sample_idx, **kwargs):
    w1 = kwargs['w1']
    b1 = kwargs['b1']
    w2 = kwargs['w2']
    b2 = kwargs['b2']

    test_num = 0
    R = rating_matrix.toarray()
    nt, mt = np.shape(R)

    whole_idx = np.arange(mt)
    cse_idx = np.setdiff1d(whole_idx, sample_idx, False)

    # sigmoid-tanh
    # temp = np.dot(w1, Vs[:, cse_idx]) + b1
    # hidden = 1/(1+(np.exp((-temp))))
    #
    # pred_Vt = np.tanh(np.dot(w2, hidden) + b2)

    # tanh-sigmoid
    # hidden = np.tanh(np.dot(w1, Vs[:, cse_idx]) + b1)
    # temp = np.dot(w2, hidden) + b2
    # pred_Vt = 1 / (1 + (np.exp((-temp))))

    # without activation function
    hidden = np.dot(w1, Vs[:, cse_idx]) + b1
    pred_Vt = np.dot(w2, hidden) + b2

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

def BPR_evaluate(Vs, Ut, sample_idx, **kwargs):
    w1 = kwargs['w1']
    b1 = kwargs['b1']
    w2 = kwargs['w2']
    b2 = kwargs['b2']

    m = kwargs['m']
    _, nt = np.shape(Ut)

    test_dict = kwargs['test_dict']
    uij = generate_batch(test_dict, nt, m)

    u, i, j = uij[:, 0] - 1, uij[:, 1] - 1, uij[:, 2] - 1
    U = Ut.T

    # sigmoid-tanh
    # temp = np.dot(w1, Vs) + b1
    # hidden = 1 / (1 + (np.exp((-temp))))
    #
    # Vt = np.tanh(np.dot(w2, hidden) + b2)
    # Vt = Vt.T

    # tanh-sigmoid
    # hidden = np.tanh(np.dot(w1, Vs) + b1)
    # temp = np.dot(w2, hidden) + b2
    # Vt = 1 / (1 + (np.exp((-temp))))
    # Vt = Vt.T

    # tanh for hidden
    # hidden = np.tanh(np.dot(w1, Vs) + b1)
    # Vt = np.dot(w2, hidden) + b2
    # Vt = Vt.T

    # sigmoid for hidden
    # temp = np.dot(w1, Vs) + b1
    # hidden = 1 / (1 + (np.exp((-temp))))
    # Vt = np.dot(w2, hidden) + b2
    # Vt = Vt.T

    # without activation function
    hidden = np.dot(w1, Vs) + b1
    Vt = np.dot(w2, hidden) + b2
    Vt = Vt.T

    u_embed = U[u, :]
    i_embed = Vt[i, :]
    j_embed = Vt[j, :]

    pr = np.sum(u_embed * (i_embed - j_embed), axis=1)
    auc = np.mean(1 * (pr > 0))

    print('AUC: %.6f' % (auc))

    return auc

if __name__ == '__main__':
    LFM = 'MF'
    if LFM == 'MF':
        source_model_path = '../LatentFactorModeling/MF/mf_s'
        source_meta_path = '../LatentFactorModeling/MF/mf_s/mf_s.ckpt.meta'

        target_model_path = '../LatentFactorModeling/MF/mf_t'
        target_meta_path = '../LatentFactorModeling/MF/mf_t/mf_t.ckpt.meta'

    elif LFM == 'BPR':
        source_model_path = '../LatentFactorModeling/BPR/bpr_s'
        source_meta_path = '../LatentFactorModeling/BPR/bpr_s/bpr_s.ckpt.meta'

        target_model_path = '../LatentFactorModeling/BPR/bpr_t'
        target_meta_path = '../LatentFactorModeling/BPR/bpr_t/bpr_t.ckpt.meta'

    # Us∈(100, |Us|), Vs∈(100, |V|)
    Us, Vs = load_embedding(source_meta_path, source_model_path)
    # Ut∈(100, |Ut|), Vt∈(100, |V|)
    Ut, Vt = load_embedding(target_meta_path, target_model_path)

    with open('./Mt.pkl', 'rb') as f:
        Mt = pickle.load(f)

    # args: input_Vs, input_Vt, alpha, lr,
        # latent_factor_model = 'MF',
        # cold_start_entities_fraction=0.2,
        # max_epoch = 10000,
        # patience_count = 5,
        # verbose=True, display_step=100
    MultiLayerPerceptron(Vs, Vt, 0.05, 0.01, rating_matrix=Mt, Ut=Ut)

