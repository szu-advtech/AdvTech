import math

import numpy as np
import os
import scipy.io as sio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import keras

# 使用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 按需动态分配显卡内存
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


place = 'houston'
# eee_name = "spectral_encoder64_encoder32_nsct64_nsct32"
# folder_path = "TEST" + place + "-" + eee_name + "-RF-"
if not os.path.exists("f1f2f3"):  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs("f1f2f3")


def softmax(x):
    x_exp = np.exp(x)
    # 如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / x_sum
    return s



def self_attention(query, key, value):
    """
    :param query: [bs, dim]
    :param key: [bs, dim]
    :param value: [bs, dim]
    :return:
    """
    bs, d_k = query.shape
    scores = np.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)

    mx = np.max(scores, axis=-1, keepdims=True)
    numerator = np.exp(scores - mx)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    self_attn = numerator / denominator
    # 计算上下文
    context = np.matmul(self_attn, value)

    return context


def f_pad(f):
    return np.pad(f, ((0, 0), (0, 144 - 128)), 'constant')


if __name__ == '__main__':
    for jj in range(14):
        for pp in range(5):
            ii = jj + 4
            train_label = sio.loadmat("label/" + place + "_encoder_train_label_ii" + str(ii + 1) + "_pp" + str(pp + 1) + ".mat")
            train_label = train_label["train_label"]
            train_label = train_label.ravel()

            # [bs, 144]
            train_data = sio.loadmat("spectral/" + place + "_spectral_train_data_ii" + str(ii + 1) + "_pp" + str(pp + 1) + ".mat")
            # [75, 144]
            f1 = train_data["train_data"]

            # [bs, 128]
            hsi32_spectral_data = sio.loadmat("hsi32/" + place + "_encoder_train_data_ii" + str(ii + 1) + "_pp" + str(pp + 1) + ".mat")
            hsi32_spectral_data = hsi32_spectral_data["train_data"]
            # [75, 144]
            f2 = f_pad(hsi32_spectral_data)

            # [75, 128]
            hsi64_spectral_data = sio.loadmat("hsi64/" + place + "_encoder_train_data_ii" + str(ii + 1) + "_pp" + str(pp + 1) + ".mat")
            hsi64_spectral_data = hsi64_spectral_data["train_data"]
            # [75, 144]
            f3 = f_pad(hsi64_spectral_data)

            # [75, 128]
            nsct32_spectral_data = sio.loadmat("nsct32/" + place + "_encoder_nsct_train_data_ii" + str(ii + 1) + "_pp" + str(pp + 1) + ".mat")
            nsct32_spectral_data = nsct32_spectral_data["train_data"]
            # [75, 144]
            f4 = f_pad(nsct32_spectral_data)

            # [75, 128]
            nsct64_spectral_data = sio.loadmat("nsct64/" + place + "_encoder_nsct_train_data_ii" + str(ii + 1) + "_pp" + str(pp + 1) + ".mat")
            nsct64_spectral_data = nsct64_spectral_data["train_data"]
            # [75, 144]
            f5 = f_pad(nsct64_spectral_data)

            # [bs, 144]
            context = self_attention(hsi32_spectral_data.transpose(-1, -2), hsi64_spectral_data.transpose(-1, -2), nsct32_spectral_data.transpose(-1, -2))
            context = context.transpose(-1, -2)
            model = RandomForestClassifier(n_estimators=300)
            model.fit(context, train_label)

            test_data = sio.loadmat("checkpoint/spectral_table_" + place + ".mat")
            spectral_table1 = test_data["spectral_table"]
            t1 = spectral_table1

            spectral_table2 = sio.loadmat("checkpoint/zsy2_checkpoint_" + place + "_best_model_sampleGAP.mat")
            spectral_table2 = spectral_table2["GAP_table"]
            t2 = f_pad(spectral_table2)

            spectral_table3 = sio.loadmat("checkpoint/zsy6_checkpoint_" + place + "_best_model_sampleGAP.mat")
            spectral_table3 = spectral_table3["GAP_table"]
            t3 = f_pad(spectral_table3)

            spectral_table4 = sio.loadmat("checkpoint/zsy2_nsct_checkpoint_" + place + "_best_model_sampleGAP.mat")
            spectral_table4 = spectral_table4["GAP_table"]
            t4 = f_pad(spectral_table4)

            spectral_table5 = sio.loadmat("checkpoint/zsy6_nsct_checkpoint_" + place + "_best_model_sampleGAP.mat")
            spectral_table5 = spectral_table5["GAP_table"]
            t5 = f_pad(spectral_table5)

            test_context = self_attention(spectral_table2, spectral_table3, spectral_table4)
            test_context = test_context
            predict_y = model.predict(test_context)
            predict_y = predict_y.reshape((predict_y.shape[0], 1))
            print(predict_y.shape)
            # sio.savemat("f1f2f3" + "/" + place + "_" + "f1f2f3" + "_train_label_ii" + str(ii + 1) + "_pp" + str(pp + 1) + "_best_model_predictY.mat", {'predict_y': predict_y})
            # print("f1f2f3" + "/" + place + "_" + "f1f2f3" + "_train_label_ii" + str(ii + 1) + "_pp" + str(pp + 1) + "_best_model_predictY.mat")
