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
eee_name = "spectral_encoder64_encoder32_nsct64_nsct32"
folder_path = "TEST" + place + "-" + eee_name + "-RF-"
if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(folder_path)

for jj in range(14):
    for pp in range(5):
        ii = jj + 4
        train_label = sio.loadmat("label/" + place + "_encoder_train_label_ii" + str(ii + 1) + "_pp" + str(pp + 1) + ".mat")
        train_label = train_label["train_label"]
        train_label = train_label.ravel()

        train_data = sio.loadmat("spectral/" + place + "_spectral_train_data_ii" + str(ii + 1) + "_pp" + str(pp + 1) + ".mat")
        train_data = train_data["train_data"]

        spectral_data = sio.loadmat("hsi32/" + place + "_encoder_train_data_ii" + str(ii + 1) + "_pp" + str(pp + 1) + ".mat")
        spectral_data = spectral_data["train_data"]
        train_data = np.hstack((train_data, spectral_data))

        spectral_data = sio.loadmat("hsi64/" + place + "_encoder_train_data_ii" + str(ii + 1) + "_pp" + str(pp + 1) + ".mat")
        spectral_data = spectral_data["train_data"]
        train_data = np.hstack((train_data, spectral_data))

        spectral_data = sio.loadmat("nsct32/" + place + "_encoder_nsct_train_data_ii" + str(ii + 1) + "_pp" + str(pp + 1) + ".mat")
        spectral_data = spectral_data["train_data"]
        train_data = np.hstack((train_data, spectral_data))

        spectral_data = sio.loadmat("nsct64/" + place + "_encoder_nsct_train_data_ii" + str(ii + 1) + "_pp" + str(pp + 1) + ".mat")
        spectral_data = spectral_data["train_data"]
        train_data = np.hstack((train_data, spectral_data))

        model = RandomForestClassifier(n_estimators=300)
        model.fit(train_data, train_label)

        test_data = sio.loadmat("checkpoint/spectral_table_" + place + ".mat")
        test_data = test_data["spectral_table"]

        spectral_table = sio.loadmat("checkpoint/zsy2_checkpoint_" + place + "_best_model_sampleGAP.mat")
        spectral_table = spectral_table["GAP_table"]
        test_data = np.hstack((test_data, spectral_table))

        spectral_table = sio.loadmat("checkpoint/zsy6_checkpoint_" + place + "_best_model_sampleGAP.mat")
        spectral_table = spectral_table["GAP_table"]
        test_data = np.hstack((test_data, spectral_table))

        spectral_table = sio.loadmat("checkpoint/zsy2_nsct_checkpoint_" + place + "_best_model_sampleGAP.mat")
        spectral_table = spectral_table["GAP_table"]
        test_data = np.hstack((test_data, spectral_table))

        spectral_table = sio.loadmat("checkpoint/zsy6_nsct_checkpoint_" + place + "_best_model_sampleGAP.mat")
        spectral_table = spectral_table["GAP_table"]
        test_data = np.hstack((test_data, spectral_table))

        predict_y = model.predict(test_data)
        predict_y = predict_y.reshape((predict_y.shape[0], 1))
        print(predict_y.shape)
        sio.savemat(folder_path + "/" + place + "_" + eee_name + "_train_label_ii" + str(ii + 1) + "_pp" + str(pp + 1) + "_best_model_predictY.mat", {'predict_y': predict_y})
        print(folder_path + "/" + place + "_" + eee_name + "_train_label_ii" + str(ii + 1) + "_pp" + str(pp + 1) + "_best_model_predictY.mat")