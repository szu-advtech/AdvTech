"""
预测分数，越低越好
"""

import numpy as np
import tf_slim as sl
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from modules.utils import extract_time
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()


def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data

    Returns:
      - predictive_score: MAE of the predictions on the original data
    """
    # Initialization on the Graph 初始化
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    # 数据量、序列长度、特征维度 stock应该是3361，24，6
    no, seq_len, dim = np.asarray(ori_data).shape

    # Set maximum sequence length and each sequence length
    # 通过函数返回最大序列长度和每个序列长度 此处time表示长度列表
    # 原始数据和生成数据的长度一样
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Builde a post-hoc RNN predictive network
    # 建立一个事后RNN预测网络
    # Network parameters  设置模型参数
    hidden_dim = int(dim / 2)   # GRU单元数
    iterations = 5000     # 训练一次模型来进行预测所需进行的迭代训练次数
    batch_size = 128

    # Input place holders  相当于定义变量，一段序列作为输入值
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len - 1, dim - 1], name="myinput_x")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")
    Y = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len - 1, 1], name="myinput_y")

    # Predictor function  预测函数
    def predictor(x, t):
        """Simple predictor function.

        Args:
          - x: time-series data  时间序列数据
          - t: time information  时间信息

        Returns:
          - y_hat: prediction  预测值
          - p_vars: predictor variables  预测变量
        """

        """
            网络结构：一个单层GRU+一个全连接层+一个sigmoid激活函数
        """
        # 定义创建层操作的上下文管理器
        with tf.compat.v1.variable_scope("predictor", reuse=tf.compat.v1.AUTO_REUSE) as vs:
            # 定义一个单层GRU网络
            # num_units表示GRU层的单元数
            p_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name='p_cell')
            # 通过GRU生成生成数据
            p_outputs, p_last_states = tf.compat.v1.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length=t)
            # 全连接层  activation_fn表示激活函数
            y_hat_logit = sl.fully_connected(p_outputs, 1, activation_fn=None)
            y_hat = tf.nn.sigmoid(y_hat_logit)
            # 如果参数名以predictor开头，则把该参数加入列表存起来
            p_vars = [v for v in tf.compat.v1.all_variables() if v.name.startswith(vs.name)]

        # 返回预测值和网络参数
        return y_hat, p_vars

    y_pred, p_vars = predictor(X, T)
    # Loss for the predictor  定义预测器的损失函数
    p_loss = tf.compat.v1.losses.absolute_difference(Y, y_pred)
    # optimizer  定义预测器的优化器
    p_solver = tf.compat.v1.train.AdamOptimizer().minimize(p_loss, var_list=p_vars)

    # Training 训练
    # Session start  session为开始标志
    sess = tf.compat.v1.Session()
    # 初始化模型参数
    sess.run(tf.compat.v1.global_variables_initializer())

    """
        Training using Synthetic dataset
            使用生成数据训练预测网络  
    """

    for itt in range(iterations):
        # Set mini-batch
        # 随机产生同数据量大小一样个数的数据
        idx = np.random.permutation(len(generated_data))
        # 按批量大小取随机产生的数据，原数据中批量号
        train_idx = idx[:batch_size]

        X_mb = list(generated_data[i][:-1, :(dim - 1)] for i in train_idx)
        T_mb = list(generated_time[i] - 1 for i in train_idx)
        Y_mb = list(
            np.reshape(generated_data[i][1:, (dim - 1)], [len(generated_data[i][1:, (dim - 1)]), 1]) for i in train_idx)

        # Train predictor
        # [p_solver, p_loss]就是要运行的函数  用feed_dict来对占位符赋值
        _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})


    # Test the trained model on the original data
    # 用原始数据测试用生成数据训练的模型
    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]   # no是原始数据所分的组数 因为要对所有原始数据进行预测，所以取所有组

    X_mb = list(ori_data[i][:-1, :(dim - 1)] for i in train_idx)
    T_mb = list(ori_time[i] - 1 for i in train_idx)
    Y_mb = list(np.reshape(ori_data[i][1:, (dim - 1)], [len(ori_data[i][1:, (dim - 1)]), 1]) for i in train_idx)

    # Prediction 获得原始数据的预测值
    pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})

    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        # 平均绝对误差MAE
        MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i, :, :])

    # 平均预测分数
    predictive_score = MAE_temp / no

    return predictive_score
