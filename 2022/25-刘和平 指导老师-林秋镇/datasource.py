import time
import numpy as np
import keras
import random
from keras.datasets import mnist
import tensorflow.python.keras.backend as K
from Paras import DATA_SPLIT_P

class DataSource(object):
    def __init__(self):
        raise NotImplementedError()

    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        raise NotImplementedError()


class Mnist(DataSource):
    IID = False

    MAX_NUM_CLASSES_PER_CLIENT = 2

    def __init__(self):

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x = np.concatenate([x_train, x_test]).astype('float32')
        self.y = np.concatenate([y_train, y_test])

        # 打乱数据的顺序
        n = self.x.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        self.x = self.x[idx]
        self.y = self.y[idx]

        # 分割数据
        data_split = DATA_SPLIT_P
        num_train = int(n * data_split[0])
        num_test = int(n * data_split[1])
        self.x_train = self.x[0:num_train]
        self.x_test = self.x[num_train:num_train + num_test]
        self.x_valid = self.x[num_train + num_test:]
        self.y_train = self.y[0:num_train]
        self.y_test = self.y[num_train:num_train + num_test]
        self.y_valid = self.y[num_train + num_test:]
        self.classes = np.unique(self.y)

        self.classes_train_idx = None
        self.classes_test_idx = None
        self.classes_valid_idx = None

    def select_classes_idx(self, y):
        classes_idx = []
        for i in range(self.classes.shape[0]):
            class_idx = np.array([j for j in range(y.shape[0]) if y[j] == self.classes[i]])
            classes_idx.append(class_idx)
        return classes_idx

    def fake_non_iid_data_with_class_train_size(self, my_class_distr, train_size,
                                                data_split=(.6, .3, .05)):
        self.classes_train_idx = None
        self.classes_test_idx = None
        self.classes_valid_idx = None

        self.classes_train_idx = self.select_classes_idx(self.y_train)
        self.classes_test_idx = self.select_classes_idx(self.y_test)
        self.classes_valid_idx = self.select_classes_idx(self.y_valid)

        test_size = int(train_size / data_split[0] * data_split[1])  # train_size * 0.214
        valid_size = int(train_size / data_split[0] * data_split[2])  # train_size * 0.214

        train_set = [self.sample_single_non_iid(self.x_train, self.y_train, my_class_distr, self.classes_train_idx)
                     for _ in range(train_size)]
        test_set_train_same = [
            self.sample_single_non_iid(self.x_test, self.y_test, my_class_distr, self.classes_test_idx)
            for _ in range(test_size)]
        test_set = [self.sample_single_non_iid(self.x_test, self.y_test, my_class_distr, self.classes_test_idx)
                    for _ in range(test_size)]
        valid_set = [self.sample_single_non_iid(self.x_valid, self.y_valid, my_class_distr, self.classes_valid_idx)
                     for _ in range(valid_size)]

        print("done generating fake data")
        return ((train_set, test_set, test_set_train_same, valid_set), my_class_distr)

    def sample_single_non_iid(self, x, y, weight=None, classes_index=None):
        chosen_class = np.random.choice(self.classes, p=weight)
        index = 0
        for i in range(self.classes.shape[0]):
            if self.classes[i] == chosen_class:
                index = i
                break
        candidates_idx = classes_index[index]
        idx = np.random.choice(candidates_idx)
        return self.post_process(x[idx], y[idx])

    def gen_dummy_non_iid_weights(self):
        self.classes = np.array(range(10))
        num_classes_this_client = random.randint(2, Mnist.MAX_NUM_CLASSES_PER_CLIENT + 1)
        classes_this_client = random.sample(self.classes.tolist(), num_classes_this_client)
        w = np.array([random.random() for _ in range(num_classes_this_client)])
        weights = np.array([0.] * self.classes.shape[0])
        for i in range(len(classes_this_client)):
            weights[classes_this_client[i]] = w[i]
        weights /= np.sum(weights)
        return weights.tolist()

    def post_process(self, xi, yi):
        # height = 224
        # width = 224
        # channels = 3
        if K.image_data_format() == 'channels_first':
            xi = xi.reshape(1, xi.shape[0], xi.shape[1])
        else:
            xi = xi.reshape(xi.shape[0], xi.shape[1], 1)
        y_vec = keras.utils.to_categorical(yi, self.classes.shape[0])
        return xi / 255., y_vec

    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        n_test = int(self.x.shape[0] * test_reserve)
        n_train = self.x.shape[0] - n_test
        nums = [n_train // num_workers] * num_workers
        nums[-1] += n_train % num_workers
        idxs = np.array([np.random.choice(np.arange(n_train), num, replace=False) for num in nums])
        return {
            "train": [self.post_process(self.x[idx], self.y[idx]) for idx in idxs],
            # (n_test * 28 * 28, n_test * 1)
            "test": self.post_process(self.x[np.arange(n_train, n_train + n_test)],
                                      self.y[np.arange(n_train, n_train + n_test)])
        }


if __name__ == "__main__":
    m = Mnist()
    my_class_distr = [0.49480815252999644, 0.0, 0.0, 0.5051918474700036, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    all_distr = [0.2328927896775581, 0.1289937864669184, 0.038995770922023804, 0.24810438185040445, 0.10366585861502065,
                 0.0, 0.09323315767177595, 0.0, 0.15411425479629864, 0.0]
    train_size = 14243
    data_split = DATA_SPLIT_P
    time_begin = time.time()
    a = m.fake_non_iid_data_with_class_train_size(my_class_distr, all_distr, train_size, data_split)
    time_end = time.time()

