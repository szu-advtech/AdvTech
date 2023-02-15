from tensorflow import keras
import pickle, uuid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
import random, codecs, json, time
from flask import *
from flask_socketio import *
import tensorflow as tf
from Paras import MIN_NUM_WORKERS_P, MAX_NUM_ROUNDS_P, NUM_CLIENTS_CONTACTED_PER_ROUND_P, \
    ROUNDS_BETWEEN_VALIDATIONS_P, DATA_TYPE, ROUNDS_BETWEEN_FULL_COMMUNICATION_P, POWER_A_P, GPU_MEMORY_FRACTION_P
from Paras import DATA_SPLIT_P, EPOCH_PER_ROUND_P, BATCH_SIZE_P, SET_VALUE
from tensorflow.python.keras.layers import deserialize

class GlobalModel(object):

    def __init__(self):
        print("进来了")
        self.shallow_model = self.build_shallow_model()
        self.shallow_current_weights = self.shallow_model.get_weights()
        self.shallow_test_accuracies_same_distri = []
        self.shallow_test_accuracies_diff_distri = []
        self.shallow_test_losses = []
        self.shallow_test_accuracies = []

        self.deep_model = self.build_deep_model()
        self.deep_current_weights = self.deep_model.get_weights()
        self.deep_test_losses = []
        self.deep_test_accuracies = []
        self.deep_test_accuracies_same_distri = []
        self.deep_test_losses_same_distri = []
        self.deep_test_accuracies_diff_distri = []

        self.prev_train_loss = None
        self.train_losses = []
        self.valid_losses = []
        self.test_accuracies = []
        self.test_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

        self.training_start_time = int(round(time.time()))

    def build_shallow_model(self):
        model = None
        if DATA_TYPE == 'MNIST':
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(10, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
        elif DATA_TYPE == 'CIFAR':
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(10, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
        elif DATA_TYPE == 'CIFAR-100':
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(100, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
        print('shallow_model_summary', model.summary())
        print('浅网络的形状是：', len(np.array(model.get_weights())))
        return model

    def build_deep_model(self):
        model = None
        if DATA_TYPE == 'MNIST':
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(10, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

        elif DATA_TYPE == 'CIFAR':
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(10, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
        elif DATA_TYPE == 'CIFAR-100':
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(100, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
        print('deep_model_summary', model.summary())
        print('深网络的形状是：', len(np.array(model.get_weights())))
        return model

    def update_shallow_weights(self, client_weights, client_sizes):
        new_weights = [np.zeros(w.shape) for w in self.shallow_current_weights]
        total_size = np.sum(client_sizes)
        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[c][i] * client_sizes[c] / int(total_size)
        self.shallow_current_weights = new_weights

    def update_deep_weights(self, client_weights, client_sizes):
        new_weights = [np.zeros(w.shape) for w in self.deep_current_weights]
        total_size = np.sum(client_sizes)
        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[c][i] * client_sizes[c] / int(total_size)
        self.deep_current_weights = new_weights

    def aggregate_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.train_losses += [[cur_round, cur_time, aggr_loss]]
        self.train_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats_AT.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def aggregate_test_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.test_losses += [[cur_round, cur_time, aggr_loss]]
        self.test_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats_AT.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.valid_losses += [[cur_round, cur_time, aggr_loss]]
        self.valid_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats_AT.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def aggregate_deep_local_test_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.deep_test_accuracies_diff_distri += [[cur_round, cur_time, aggr_loss]]
        self.deep_test_accuracies_same_distri += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def aggregate_shallow_local_test_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.shallow_test_accuracies_diff_distri += [[cur_round, cur_time, aggr_loss]]
        self.shallow_test_accuracies_same_distri += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def aggregate_deep_test_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.deep_test_losses += [[cur_round, cur_time, aggr_loss]]
        self.deep_test_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def aggregate_shallow_test_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.shallow_test_losses += [[cur_round, cur_time, aggr_loss]]
        self.shallow_test_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies,
            "test_accuracy": self.test_accuracies,
            "deep_test_loss": self.deep_test_losses,  # by_cy
            "deep_test_accuracy": self.deep_test_accuracies,
            "shallow_test_loss": self.shallow_test_losses,  # by_cy
            "shallow_test_accuracy": self.shallow_test_accuracies,
            "deep_test_accuracy_same_distri": self.deep_test_accuracies_same_distri,
            "deep_test_accuracies_diff_distri": self.deep_test_accuracies_diff_distri,
            "shallow_test_accuracy_same_distri": self.shallow_test_accuracies_same_distri,
            "shallow_test_accuracy_diff_distri": self.shallow_test_accuracies_diff_distri,
            # by_cy
        }


def aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes):
    total_size = np.sum(client_sizes)
    aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                       for i in range(len(client_sizes)))
    aggr_accuraries = np.sum(client_accuracies[i] / total_size * client_sizes[i]
                             for i in range(len(client_sizes)))
    return aggr_loss, aggr_accuraries


class FLServer(object):

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"  # （代表仅使用第0，1号GPU）

    config = tf.compat.v1.ConfigProto()

    config.gpu_options.allow_growth = False  # by_cy_gpu
    config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION_P  # by_cy_gpu
    sess = tf.compat.v1.Session(config=config)

    MIN_NUM_WORKERS = MIN_NUM_WORKERS_P  # 最少需要多少client
    MAX_NUM_ROUNDS = MAX_NUM_ROUNDS_P
    NUM_CLIENTS_CONTACTED_PER_ROUND = NUM_CLIENTS_CONTACTED_PER_ROUND_P  # 每轮通信选择多少个client
    ROUNDS_BETWEEN_VALIDATIONS = ROUNDS_BETWEEN_FULL_COMMUNICATION_P[0]
    POWER_A = POWER_A_P

    #########################################################################################################
    ROUNDS_BETWEEN_FULL_COMMUNICATION = ROUNDS_BETWEEN_FULL_COMMUNICATION_P

    #########################################################################################################

    def __init__(self, host, port):
        self.global_model = GlobalModel()
        self.ready_client_sids = {}

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port

        self.current_round = -1

        self.client_updates_transfer = []
        self.current_round_client_updates = []
        self.shallow_weights = []
        self.deep_weights = []
        self.deep_train_loss = []
        self.shallow_train_loss = []
        self.deep_size = []
        self.shallow_size = []
        self.eval_client_updates = []
        self.deep_acc = []
        self.deep_loss = []
        self.deep_acc_same_distr = []
        self.deep_loss_same_distr = []
        self.deep_test_size = []
        self.deep_local_test_size = []
        self.shallow_acc = []
        self.shallow_loss = []
        self.shallow_acc_same_distr = []
        self.shallow_loss_same_distr = []
        self.shallow_test_size = []
        self.shallow_local_test_size = []
        self.deep_test_loss = []
        self.deep_local_test_accuracy = []
        self.shallow_local_test_loss = []
        self.shallow_local_test_accuracy = []
        self.weight_states_dic = None
        self.weight_states_transfer_dic = None

        self.shallow_rid = list()
        self.deep_rid = list()

        self.register_handles()

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())

    def register_handles(self):

        @self.socketio.on('connect')
        def handle_connect():
            print("{}   connected".format(request.sid))

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print("{}   reconnected".format(request.sid))

        @self.socketio.on('disconnect')
        def handle_reconnect():
            print("{}   disconnected".format(request.sid))
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.pop(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up(data):
            print("client wake_up: {}".format(request.sid), 'train_size:', data['train_size'])
            if data['train_size'] <= SET_VALUE:
                current_model = self.global_model.shallow_model
            else:
                current_model = self.global_model.deep_model
            emit('init', {
                'train_size': data['train_size'],
                'shallow_model_json': self.global_model.shallow_model.to_json(),
                'deep_model_json': self.global_model.deep_model.to_json(),
                'model_json': current_model.to_json(),
                'data_split': DATA_SPLIT_P,
                'epoch_per_round': EPOCH_PER_ROUND_P,
                'batch_size': BATCH_SIZE_P
            })

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            print("client ready for training: {}, {}".format(request.sid, data))
            self.ready_client_sids.update({request.sid: data['train_size']})
            if len(self.ready_client_sids) >= FLServer.MIN_NUM_WORKERS and self.current_round == -1:
                self.train_next_round()  # MIN_NUM_WORKERS是最小的客户机数量，所以要所有的客户机都准备好了才开始训练

        @self.socketio.on('client_update')
        def handle_client_update(data):
            print("received client update of bytes: {} \nhandle client_update: {}".format(sys.getsizeof(data),
                                                                                          request.sid))

            if data['round_number'] == self.current_round:
                # 这个判断应该是为了设置同步
                self.current_round_client_updates += [data]
                # 设置同步
                if len(self.current_round_client_updates) == FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND:
                    for x in self.current_round_client_updates:
                        if len(pickle_string_to_obj(x['weights'])) == len(self.global_model.deep_current_weights):
                            self.deep_weights.append(pickle_string_to_obj(x['weights']))
                            self.deep_size.append(x['train_size'])
                            self.deep_train_loss.append(x['train_loss'])

                            if data['round_number'] % ROUNDS_BETWEEN_VALIDATIONS_P == 1 and data['round_number'] != 1:
                                self.deep_acc.append(x['test_accuracy'])
                                self.deep_acc_same_distr.append(x['test_accuracy_same'])
                                self.deep_local_test_size.append(x['test_size'])

                        else:
                            self.shallow_weights.append(pickle_string_to_obj(x['weights']))
                            self.shallow_size.append(x['train_size'])
                            self.shallow_train_loss.append(x['train_loss'])

                            if data['round_number'] % ROUNDS_BETWEEN_VALIDATIONS_P == 1 and data['round_number'] != 1:
                                self.shallow_acc.append(x['test_accuracy'])
                                self.shallow_acc_same_distr.append(x['test_accuracy_same'])
                                self.shallow_local_test_size.append(x['test_size'])

                    ####################################### 模型聚合 ##############################################
                    self.global_model.update_deep_weights(
                        [x for x in self.deep_weights],
                        [size for size in self.deep_size],
                    )

                    self.global_model.update_shallow_weights(
                        [x for x in self.shallow_weights],
                        [size for size in self.shallow_size],
                    )

                    if data['round_number'] % ROUNDS_BETWEEN_VALIDATIONS_P == 1 and data['round_number'] != 1:
                        aggr_deep_test_loss_same_distr, aggr_deep_test_accuracy_same_distr = self.global_model.aggregate_deep_local_test_loss_accuracy(
                            [x for x in self.deep_acc],
                            [x for x in self.deep_acc_same_distr],
                            [x for x in self.deep_local_test_size],
                            self.current_round
                        )
                        aggr_shallow_test_loss_distr, aggr_shallow_test_accuracy_same_distr = self.global_model.aggregate_shallow_local_test_loss_accuracy(
                            [x for x in self.shallow_acc],
                            [x for x in self.shallow_acc_same_distr],
                            [x for x in self.shallow_local_test_size],
                            self.current_round
                        )

                    ####################################### 模型准确率和损失 ##############################################
                    aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
                        [x['train_loss'] for x in self.current_round_client_updates],
                        [x['train_accuracy'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                        self.current_round
                    )

                    if 'valid_loss' in self.current_round_client_updates[0]:
                        aggr_valid_loss, aggr_valid_accuracy = self.global_model.aggregate_valid_loss_accuracy(
                            [x['valid_loss'] for x in self.current_round_client_updates],
                            [x['valid_accuracy'] for x in self.current_round_client_updates],
                            [x['valid_size'] for x in self.current_round_client_updates],
                            self.current_round
                        )

                        self.valid_and_eval()

                    self.global_model.prev_train_loss = aggr_train_loss

                    if self.current_round >= FLServer.MAX_NUM_ROUNDS:
                        self.stop_and_eval()
                    else:
                        self.train_next_round()

        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            if self.eval_client_updates is None:  # []也不算是None
                return
            self.eval_client_updates += [data]
            if len(self.eval_client_updates) == len(self.ready_client_sids):
                # 这里和下面都是计算平均的loss和accuracy的，算法的计算方式都相当（调用同一个模块），只是写入的东西不一样
                for x in self.eval_client_updates:
                    if x['train_size'] <= SET_VALUE:
                        self.shallow_acc.append(x['test_accuracy'])
                        self.shallow_loss.append(x['test_loss'])
                        self.shallow_test_size.append(x['test_size'])
                        print("test_size:", x['test_size'], " test_accuracy", x['test_accuracy'], " test_loss",
                              x['test_loss'])
                    else:
                        self.deep_acc.append(x['test_accuracy'])
                        self.deep_loss.append(x['test_loss'])
                        self.deep_test_size.append(x['test_size'])
                        print("test_size:", x['test_size'], " test_accuracy", x['test_accuracy'], " test_loss",
                              x['test_loss'])

                deep_aggr_test_loss, deep_aggr_test_accuracy = self.global_model.aggregate_deep_test_loss_accuracy(
                    [x for x in self.deep_loss],
                    [y for y in self.deep_acc],
                    [size for size in self.deep_test_size],
                    self.current_round
                )
                shallow_aggr_test_loss, shallow_aggr_test_accuracy = self.global_model.aggregate_shallow_test_loss_accuracy(
                    [x for x in self.shallow_loss],
                    [y for y in self.shallow_acc],
                    [size for size in self.shallow_test_size],
                    self.current_round
                )

                print(
                    "\ndeep_aggr_test_loss: {} \ndeep_aggr_test_accuracy: {} \n##########== overall_test ==##########".format(
                        deep_aggr_test_loss, deep_aggr_test_accuracy))
                print(
                    "\nshallow_aggr_test_loss: {} \nshallow_aggr_test_accuracy: {} \n##########== overall_test ==##########".format(
                        shallow_aggr_test_loss, shallow_aggr_test_accuracy))
                self.eval_client_updates = []

    def obtain_client_updates_a(self):
        self.client_updates = []
        for i in self.weight_states_dic.values():
            dis_round = self.current_round - i['round_stamp']  # 这里应该全是0呀
            k = FLServer.POWER_A ** (-dis_round)  # FlServer.POWER_A=e，k是全1
            i['data'].update({'train_size_update_weights': k * i['data']['train_size']})
            self.client_updates += [i['data']]  # 在本来的weight_states_dic上增加了一个变量train_size_update_weights
            if dis_round == 0 and 'valid_loss' in i['data']:  # 一定要是5的倍数
                self.current_round_client_updates += [i['data']]  # current_round_client_updates只记录5的倍数次数

    def obtain_client_updates_transfer_a(self):
        self.client_updates_transfer = []
        for i in self.weight_states_transfer_dic.values():
            dis_round = self.current_round - i['round_stamp']
            k = FLServer.POWER_A ** (-dis_round)
            i['data'].update({'train_size_update_weights': k * i['data']['train_size']})
            self.client_updates_transfer += [i['data']]

    def train_next_round(self):

        self.current_round_client_updates = []
        self.shallow_weights = []
        self.deep_weights = []
        self.deep_size = []
        self.shallow_size = []
        self.deep_train_loss = []
        self.shallow_train_loss = []
        self.deep_local_test_loss = []
        self.deep_local_test_accuracy = []
        self.shallow_local_test_loss = []
        self.shallow_local_test_accuracy = []
        self.deep_acc = []
        self.deep_acc_same_distr = []
        self.deep_local_test_size = []
        self.shallow_acc = []
        self.shallow_acc_same_distr = []
        self.shallow_local_test_size = []

        self.current_round += 1
        self.shallow_rid = list()
        self.deep_rid = list()

        print("### Round: {} ###".format(self.current_round))

        client_sids_selected = random.sample(list(self.ready_client_sids), FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND)
        # NUM_CLIENTS_CONTACTED_PER_ROUND=5，取5个来训练咯
        print("request updates from: {}".format(client_sids_selected))

        for rid in client_sids_selected:
            if self.ready_client_sids.get(rid) <= SET_VALUE:
                self.shallow_rid.append(rid)
            else:
                self.deep_rid.append(rid)

        print("############## Round", self.current_round, "shallow_rid: ", self.shallow_rid, "########################")
        print("################# Round", self.current_round, "deep_rid: ", self.deep_rid, "###########################")

        for rid in self.shallow_rid:
            emit('request_update', {
                'ready_client_sids': rid,
                'round_number': self.current_round,
                'current_weights': obj_to_pickle_string(self.global_model.shallow_current_weights),
                'weights_format': 'pickle',
                'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
            }, room=rid)
        for rid in self.deep_rid:
            emit('request_update', {
                'ready_client_sids': rid,
                'round_number': self.current_round,
                'current_weights': obj_to_pickle_string(self.global_model.deep_current_weights),
                'weights_format': 'pickle',
                'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
            }, room=rid)

    def valid_and_eval(self):
        self.eval_client_updates = []
        self.deep_acc = []
        self.deep_loss = []
        self.deep_acc_same_distr = []
        self.deep_loss_same_distr = []
        self.deep_test_size = []
        self.shallow_acc = []
        self.shallow_loss = []
        self.shallow_acc_same_distr = []
        self.shallow_loss_same_distr = []
        self.shallow_test_size = []

        for rid in self.shallow_rid:
            emit('valid_and_eval', {
                'round_number': self.current_round,
                'ready_client_sids': rid,
                'current_weights': obj_to_pickle_string(self.global_model.shallow_current_weights),
                'weights_format': 'pickle'
            }, room=rid)

        for rid in self.deep_rid:
            emit('valid_and_eval', {
                'round_number': self.current_round,
                'ready_client_sids': rid,
                'current_weights': obj_to_pickle_string(self.global_model.deep_current_weights),
                'weights_format': 'pickle'
            }, room=rid)

    def stop_and_eval(self):
        self.deep_acc = []
        self.deep_loss = []
        self.deep_acc_same_distr = []
        self.deep_loss_same_distr = []
        self.deep_test_size = []
        self.shallow_acc = []
        self.shallow_loss = []
        self.shallow_acc_same_distr = []
        self.shallow_loss_same_distr = []
        self.shallow_test_size = []
        self.eval_client_updates = []

        for rid in self.shallow_rid:
            emit('stop_and_eval', {
                'round_number': self.current_round,
                'ready_client_sids': rid,
                'current_weights': obj_to_pickle_string(self.global_model.shallow_current_weights),
                'weights_format': 'pickle'
            }, room=rid)

        for rid in self.deep_rid:
            emit('stop_and_eval', {
                'round_number': self.current_round,
                'ready_client_sids': rid,
                'current_weights': obj_to_pickle_string(self.global_model.deep_current_weights),
                'weights_format': 'pickle'
            }, room=rid)

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))


def set_weights(model, new_weights):
    model.set_weights(new_weights)
    return model


if __name__ == '__main__':
    server = FLServer("127.0.0.1", 5000)

    server.start()
    print("listening on 127.0.0.1:5000")
