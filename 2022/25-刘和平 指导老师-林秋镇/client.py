from Paras import DATA_TYPE, ROUNDS_BETWEEN_VALIDATIONS_P, large_test_distr, small_test_distr, \
    SET_VALUE, GPU_MEMORY_FRACTION_P, SVD_INDEX_DEEP_1, SVD_INDEX_DEEP_2
import numpy as np
import pickle
from socketIO_client import SocketIO, LoggingNamespace
import datasource
import random, codecs, json, time
import copy
from tensorflow import keras
from keras.models import model_from_json

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "4"  # （代表仅使用第0，1号GPU）


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))


class LocalModel(object):
    def __init__(self, model_config, data_collected):
        self.model_config = model_config
        self.model = model_from_json(model_config['model_json'])
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

        train_data, test_data, test_data_same_train_distr_, valid_data = data_collected
        self.x_train = np.array([tup[0] for tup in train_data])
        self.y_train = np.array([tup[1] for tup in train_data])

        self.x_test = np.array([tup[0] for tup in test_data])
        self.y_test = np.array([tup[1] for tup in test_data])

        self.x_valid = np.array([tup[0] for tup in valid_data])
        self.y_valid = np.array([tup[1] for tup in valid_data])

        self.x_test_train_distr = np.array([tup[0] for tup in test_data_same_train_distr_])
        self.y_test_train_distr = np.array([tup[1] for tup in test_data_same_train_distr_])

        if DATA_TYPE == 'CIFAR':
            self.y_test = np.reshape(self.y_test, [len(self.y_test), 10])
            self.y_valid = np.reshape(self.y_valid, [len(self.y_valid), 10])
            self.y_train = np.reshape(self.y_train, [len(self.y_train), 10])
            self.y_test_train_distr = np.reshape(self.y_test_train_distr, [len(self.y_test_train_distr), 10])
        if DATA_TYPE == 'CIFAR_100':
            self.y_test = np.reshape(self.y_test, [len(self.y_test), 100])
            self.y_valid = np.reshape(self.y_valid, [len(self.y_valid), 100])
            self.y_train = np.reshape(self.y_train, [len(self.y_train), 100])
            self.y_test_train_distr = np.reshape(self.y_test_train_distr, [len(self.y_test_train_distr), 100])

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    def train_one_round(self):
        print("开始训练", self.x_train.size / 28 / 28)
        self.model.fit(self.x_train, self.y_train,
                       epochs=self.model_config['epoch_per_round'],
                       batch_size=self.model_config['batch_size'],
                       verbose=0,
                       validation_split=0.0, validation_data=None)
        print("训练好了", self.x_train.size / 28 / 28)
        score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return self.model.get_weights(), score[0], score[1]

    def validate(self):
        score = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)

        return score

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return score

    def evaluate_same_distr(self):
        score = self.model.evaluate(self.x_test_train_distr, self.y_test_train_distr, verbose=0)
        return score

    def reverse_SVD(self, u, s, vt):
        S = np.zeros([len(u), len(vt)])
        S[:len(s), :len(s)] = np.diag(s)
        results = u.dot(S.dot(vt))
        return results


class FederatedClient(object):

    def __init__(self, value_data_class_train_size, server_host, server_port, datasource):
        self.local_model = None
        self.my_class_distr = value_data_class_train_size['class_distr']
        self.train_size = value_data_class_train_size['train_size']

        if self.train_size >= SET_VALUE:
            self.test_distr = large_test_distr
        else:
            self.test_distr = small_test_distr

        self.datasource = datasource()
        self.sio = SocketIO(server_host, server_port, LoggingNamespace)
        self.register_handles()
        print('sent wakeup')
        self.sio.emit('client_wake_up', {'train_size': self.train_size})
        self.sio.wait()

        self.u_1 = None
        self.vt_1 = None
        self.u_2 = None
        self.vt_2 = None

    def on_init(self, *args):
        model_config = args[0]
        fake_data, my_class_distr = self.datasource.fake_non_iid_data_with_class_train_size(self.my_class_distr,
                                                                                            self.test_distr,
                                                                                            self.train_size,
                                                                                            data_split=model_config[
                                                                                                'data_split'])

        self.local_model = LocalModel(model_config, fake_data)
        self.sio.emit('client_ready', {
            'train_size': self.local_model.x_train.shape[0],
            'class_distr': my_class_distr,
        })
        print('已发送client_ready')

    def register_handles(self):
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_request_update(*args):
            req = args[0]
            print("######################update requested Round =", req['round_number'],
                  "#############################")
            weights = None
            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])

            #################################### 对模型进行更新 ################################################
            # req['round_number'] % 5 == 0 是上传的特殊轮
            # 下载的特殊轮是 req['round_number'] % 5 == 1 ，下载特殊轮就直接全部更新
            if req['round_number'] == 0:
                self.local_model.set_weights(weights)
            else:
                # 如果是特殊轮，这不用奇异值矩阵
                if req['round_number'] % ROUNDS_BETWEEN_VALIDATIONS_P == 1 and req['round_number'] != 1:
                    print("进行的是不用SVD的下载")
                    self.local_model.set_weights(weights)
                else:
                    # 特殊轮的话要对奇异值矩阵进行处理
                    print("进行的是用SVD的下载")
                    rev_weights_1 = self.local_model.reverse_SVD(self.u_1, weights[SVD_INDEX_DEEP_1][0], self.vt_1)
                    rev_weights_2 = self.local_model.reverse_SVD(self.u_2, weights[SVD_INDEX_DEEP_2][0], self.vt_2)
                    temp_weights = copy.deepcopy(weights)
                    temp_weights[SVD_INDEX_DEEP_1] = rev_weights_1
                    temp_weights[SVD_INDEX_DEEP_2] = rev_weights_2
                    self.local_model.set_weights(temp_weights)

            # 还没进行本地运算的模型，如果是特殊轮，还要算一下这些
            if req['round_number'] % ROUNDS_BETWEEN_VALIDATIONS_P == 1:
                local_test_loss, local_test_accuracy = self.local_model.evaluate()
                # print('aggre__local_test_acc', req['ready_client_sids'], local_test_accuracy)
                test_loss_same_distr, test_accuracy_same_distr = self.local_model.evaluate_same_distr()
                # print('aggre_local_test_acc_same', req['ready_client_sids'], test_accuracy_same_distr)

            model_weights, train_loss, train_accuracy = self.local_model.train_one_round()
            print("train_size:", self.train_size, "train_loss", train_loss, "train_accuracy", train_accuracy)
            # print('successful train weights')
            self.u_1, sigma_1, self.vt_1 = np.linalg.svd(weights[SVD_INDEX_DEEP_1])
            self.u_2, sigma_2, self.vt_2 = np.linalg.svd(weights[SVD_INDEX_DEEP_2])

            upload_weights = copy.deepcopy(weights)

            # 进行了本地计算的模型，如果是特殊轮，再算一遍这些
            if req['round_number'] % ROUNDS_BETWEEN_VALIDATIONS_P == 1:
                test_loss, test_accuracy = self.local_model.evaluate()
                # print('local_test_acc', req['ready_client_sids'], test_accuracy)

            #################################### 对模型进行上传 ################################################
            # 判断是不是特殊轮的条件：req['round_number'] % 5 == 0,如果符合条件,则要进行SVD的上传
            if req['round_number'] % ROUNDS_BETWEEN_VALIDATIONS_P == 0 and req['round_number'] != 0:
                print("上传的是不含SVD的矩阵")
                my_weights = copy.deepcopy(upload_weights)
            else:
                print("上传的是含SVD的矩阵")
                upload_weights[SVD_INDEX_DEEP_1] = sigma_1
                upload_weights[SVD_INDEX_DEEP_2] = sigma_1
                my_weights = copy.deepcopy(upload_weights)

            resp = {
                'round_number': req['round_number'],
                'weights': obj_to_pickle_string(my_weights),
                'train_size': self.local_model.x_train.shape[0],
                'valid_size': self.local_model.x_valid.shape[0],
                'train_loss': float(train_loss),
                'train_accuracy': float(train_accuracy),
            }
            if req['round_number'] % ROUNDS_BETWEEN_VALIDATIONS_P == 1:
                resp['test_accuracy'] = float(local_test_accuracy)
                resp['test_accuracy_same'] = float(test_accuracy_same_distr)
                resp['test_size'] = self.local_model.x_test.shape[0]

            if req['run_validation']:
                valid_loss, valid_accuracy = self.local_model.validate()
                resp['valid_loss'] = float(valid_loss)
                resp['valid_accuracy'] = float(valid_accuracy)
            self.sio.emit('client_update', resp)

        def on_valid_and_eval(*args):
            req = args[0]
            weights = None
            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])
            self.local_model.set_weights(weights)

            test_loss, test_accuracy = self.local_model.evaluate()
            print('test_size:', self.local_model.x_test.shape[0], "test_loss:", test_loss, "test_accuracy:",
                  test_accuracy)
            test_loss_same_distr, test_accuracy_same_distr = self.local_model.evaluate_same_distr()
            # print('aggre_local_test_acc', req['ready_client_sids'], test_accuracy)
            resp = {
                'train_size': self.train_size,
                'test_size': self.local_model.x_test.shape[0],
                'test_loss': float(test_loss),
                'test_accuracy': float(test_accuracy),
                'test_loss_same_distr': float(test_loss_same_distr),
                'test_accuracy_same_distr': float(test_accuracy_same_distr)
            }
            self.sio.emit('client_eval', resp)

        def on_stop_and_eval(*args):
            req = args[0]
            weights = None
            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])

            if req['round_number'] == 0:
                self.local_model.set_weights(weights)
                print("执行了我认为不可能的语句")
            else:
                rev_weights_1 = self.local_model.reverse_SVD(self.u_1, weights[SVD_INDEX_DEEP_1][0], self.vt_1)
                rev_weights_2 = self.local_model.reverse_SVD(self.u_2, weights[SVD_INDEX_DEEP_2][0], self.vt_2)
                temp_weights = copy.deepcopy(weights)
                temp_weights[SVD_INDEX_DEEP_1] = rev_weights_1
                temp_weights[SVD_INDEX_DEEP_2] = rev_weights_2
                self.local_model.set_weights(temp_weights)

            test_loss, test_accuracy = self.local_model.evaluate()
            resp = {
                'test_size': self.local_model.x_test.shape[0],
                'test_loss': float(test_loss),
                'test_accuracy': float(test_accuracy)
            }
            self.sio.emit('client_eval', resp)

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', lambda *args: self.on_init(*args))
        self.sio.on('request_update', on_request_update)
        self.sio.on('valid_and_eval', on_valid_and_eval)
        self.sio.on('stop_and_eval', on_stop_and_eval)

    def intermittently_sleep(self, p=.1, low=10, high=36):
        if random.random() < p:
            time.sleep(random.randint(low, high))


if __name__ == "__main__":
    # FederatedClient("127.0.0.1", 3000, datasource.Mnist)
    value_data_class_train_size = {'class_distr': [0.21198145753809364, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7880185424619064, 0.0],'train_size': 4000}
    c = FederatedClient(value_data_class_train_size, "127.0.0.1", 5000, datasource.Mnist)
