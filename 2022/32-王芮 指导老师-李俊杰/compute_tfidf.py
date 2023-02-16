import math
from scipy import sparse
import numpy as np
import os
import file_handling as fh


def get_tfidf(dir, prefix):
    path = os.path.join(dir, prefix + '.tfidf.npz')
    tfidf = fh.load_sparse(path).tocsr()
    print("get tfidf data successfully")
    return tfidf

def set_tfidf(dir, prefix):
    tfidf_path = os.path.join(dir, prefix + '.tfidf.npz')
    flag = os.path.exists(tfidf_path)
    if not flag:
        compute_tfidf(dir, prefix)

    print("compute tfidf data successfully")

def compute_tfidf(input_dir, train_prefix):

    test_prefix = 'test'
    train_X, train_data, train_row, train_col, train_shape = load_input_data(input_dir, train_prefix)
    train_X = train_X.todense()
    test_X, test_data, test_row, test_col, test_shape = load_input_data(input_dir, test_prefix)
    test_X = test_X.todense()

    X = np.append(train_X, test_X, axis=0)
    data = np.append(train_data, test_data, axis=0)
    row = np.append(train_row, test_row, axis=0)
    col = np.append(train_col, test_col, axis=0)

    shape = test_shape
    shape[0] = train_shape[0] + test_shape[0]
    shape[1] = train_shape[1]

    # 每个词的tfidf值最后放在tfidf_data中
    tfidf_data = np.zeros_like(data, dtype=float)
    # 数据长度
    data_len = len(data)
    # 文本数量
    doc_num = shape[0]
    # 词汇表长度
    vocab_num = shape[1]
    # 每篇文章中的词数量
    doc_count = np.zeros(doc_num, dtype=float)

    # 每个词出现的文章数量
    word_count = np.zeros(vocab_num, dtype=float)

    j = 0
    # 计算每篇文章中的词数量和每个词出现的文章数量
    for i in range(data_len):
        if row[i] == j or row[i] + train_shape[0] == j:
            doc_count[j] += data[i]
        else:
            j = j + 1
            doc_count[j] += data[i]
        word_count[col[i]] += 1

    # 计算tf和idf
    word_tf = np.zeros_like(X, dtype=float)
    word_idf = np.zeros(vocab_num, dtype=float)
    for i in range(data_len):
        word_tf[row[i], col[i]] = data[i] / doc_count[row[i]]
        word_idf[col[i]] = math.log(doc_num / word_count[col[i]])

        # tfidf = tf * idf
        tfidf_data[i] = word_tf[row[i], col[i]] * word_idf[col[i]]

    train_tfidf = tfidf_data[0:len(train_data)]
    test_tfidf = tfidf_data[len(train_data):]
    train_tfidf_path = os.path.join(input_dir, train_prefix + '.tfidf.npz')
    test_tfidf_path = os.path.join(input_dir, test_prefix + '.tfidf.npz')

    save_tfidf(train_tfidf_path, train_tfidf, train_row, train_col, train_shape)
    save_tfidf(test_tfidf_path, test_tfidf, test_row, test_col, test_shape)


def save_tfidf(path, tfidf, row, col, shape):
    # 转为稀疏矩阵
    tfidf = sparse.coo_matrix((tfidf, (row, col)), shape=shape)
    # 保存tfidf值
    sparse.save_npz(path, tfidf)

# 载入原始数据，分别获取X中的每一列值用来求对应的tfidf值
def load_input_data(input_dir, input_prefix):

    # load the word counts and convert to a dense matrix
    # temp = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).todense()
    # X = np.array(temp, dtype='float32')
    npy = np.load(os.path.join(input_dir, input_prefix + '.npz'))
    data = npy['data']
    row = npy['row']
    col = npy['col']
    shape = npy['shape']
    X = sparse.coo_matrix((data, (row, col)), shape=shape).tocsc()
    return X, data, row, col, shape


