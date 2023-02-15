"""
判别分数，越低越好
"""

import numpy as np
import tf_slim as sl
import tensorflow as tf

from sklearn.metrics import accuracy_score
from modules.utils import train_test_divide, extract_time, batch_generator
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()


def discriminative_score_metrics(ori_data, generated_data):
    """
    Use post-hoc RNN to classify original data and synthetic data
    使用一个事后RNN对原始数据和生成数据进行分类

    Args:
        - ori_data: original data  原始数据
        - generated_data: generated synthetic data  生成数据

    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)  鉴别分数
    """
    # Initialization on the Graph
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Set maximum sequence length and each sequence length
    # ori_time是生成数据的时间信息，其实就是每组数据的长度 stock为24
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Build a post-hoc RNN discriminator network
    # Network parameters
    hidden_dim = int(dim/2)
    iterations = 2000
    batch_size = 128

    # Input place holders
    '''
        Feature
        X : 原始数据
        X_hat : 生成数据
        T : 原始时间信息
        T_hat : 生成时间信息
    '''
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    X_hat = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x_hat")

    T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")
    T_hat = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t_hat")

    # 判别器
    def discriminator(x, t):
        """
        Simple discriminator function.
        Args:
            - x: time-series data  时间序列数据
            - t: time information  时间信息

        Returns:
            - y_hat_logit: logits of the discriminator output  判别器输出结果的对数
            - y_hat: discriminator output  判别器输出结果
            - d_vars: discriminator variables  判别器变量
        """
        # reuse = tf.compat.v1.AUTO_REUSE 表示共享变量
        with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE) as vs:
            d_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name='d_cell')
            # 创建由指定cell创建的RNN
            d_outputs, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length=t)
            # Fully connected layers built on the last state rather than the output
            y_hat_logit = sl.fully_connected(d_last_states, 1, activation_fn=None)
            y_hat = tf.nn.sigmoid(y_hat_logit)
            d_vars = [v for v in tf.compat.v1.all_variables() if v.name.startswith(vs.name)]  # 模型的变量

        return y_hat_logit, y_hat, d_vars

    # 原始数据经过模型后的预测值
    y_logit_real, y_pred_real, d_vars = discriminator(X, T)
    # 生成数据经过模型后的预测值
    y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)

    # Loss for the discriminator  reduce_mean：对结果进行降维并计算损失平均值
    # 损失函数为交叉熵损失函数
    # 原始数据的损失
    d_loss_real = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit_real,
                                                                       labels=tf.ones_like(y_logit_real)))
    # 生成数据的损失
    d_loss_fake = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit_fake,
                                                                       labels=tf.zeros_like(y_logit_fake)))
    d_loss = d_loss_real + d_loss_fake

    # optimizer  变量更新
    d_solver = tf.compat.v1.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)

    # Train the discriminator
    # Start session and initialize
    sess = tf.compat.v1.Session()
    # 全局变量初始化
    sess.run(tf.compat.v1.global_variables_initializer())

    # Train/test division for both original and generated data
    # 对原始和生成数据划分训练集和测试集
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
    train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # Training step
    for itt in range(iterations):
        # Batch setting 批量获取数据 因为不批量训练，太耗资源
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

        # Train discriminator
        _, step_d_loss = sess.run([d_solver, d_loss],
                                  feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})

    # Test the performance on the testing set 对测试集进行测试
    y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake],
                                                  feed_dict={X: test_x, T: test_t,
                                                             X_hat: test_x_hat, T_hat: test_t_hat})

    # np.squeeze函数用来删除数组中维数为1的维度删掉
    y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr), ]), np.zeros([len(y_pred_fake_curr), ])), axis=0)

    # Compute the accuracy
    # accuracy_score 分类准确率（分类正确的百分比）
    # 用0.5来作为原始数据和生成数据划分的杠杆 >0.5为原始数据置1 <0.5为生成数据置0
    acc = accuracy_score(y_label_final, (y_pred_final>0.5))
    discriminative_score = np.abs(0.5-acc)
    return discriminative_score

