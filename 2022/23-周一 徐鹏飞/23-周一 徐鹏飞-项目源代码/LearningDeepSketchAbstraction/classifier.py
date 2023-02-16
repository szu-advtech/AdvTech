import os
from argparse import ArgumentParser

import keras_preprocessing.sequence
from colorama import Fore
import platform

if platform.system() == "Windows":  # for my machine
    from keras.utils import Sequence
else:  # for the server
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from tensorflow.keras.utils import Sequence

import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import models
from keras import layers
from keras.callbacks import CSVLogger
from keras.utils.np_utils import to_categorical
import dataset
import classifier_config
import abs_utils
import global_config


# Note: classifier
classifier_model: keras.Model = None  # models.Sequential()

epoch_shuffle_seeds = abs_utils.epoch_shuffle_seed(classifier_config.shuffle_seed(), classifier_config.epoch())
epoch_shuffle_seeds_idx: int = 0

__EVALUATION_MODE__: bool = False
__TRAIN_AND_EVALUATE_:bool = False


class OnEpochEnd(keras.callbacks.Callback):
    """
    Workaround to the bugged Sequence.on_epoch_end
    """

    def __init__(self):
        return

    def on_epoch_end(self, epoch, logs=None):
        global epoch_shuffle_seeds, epoch_shuffle_seeds_idx

        current_epoch: int = epoch_shuffle_seeds_idx

        seed = epoch_shuffle_seeds[epoch_shuffle_seeds_idx]
        epoch_shuffle_seeds_idx = (epoch_shuffle_seeds_idx + 1) % len(epoch_shuffle_seeds)

        dataset.shuffle_train_data(seed)

        save_filepath: str = classifier_config.classifier_model_path() + "epoch_{}/".format(current_epoch)
        global classifier_model  # classifier is not None, ignore the warning
        classifier_model.save(filepath=save_filepath,
                              overwrite=True, include_optimizer=True)
        return


class TrainDataSequence(Sequence):
    """
    Training data sequence generator.
    -> Give shuffled train data sequence.
    Note: `Train Data Sequence` and `Test Data Sequence` cannot be used on the same run.
    """

    def __init__(self):
        self.num_batches = -1
        global epoch_shuffle_seeds, epoch_shuffle_seeds_idx
        seed = epoch_shuffle_seeds[epoch_shuffle_seeds_idx]
        epoch_shuffle_seeds_idx = (epoch_shuffle_seeds_idx + 1) % len(epoch_shuffle_seeds)

        dataset.shuffle_train_data(seed)
        return

    def __len__(self):
        if self.num_batches == -1:
            self.num_batches = classifier_config.num_training_per_category() \
                               * len(dataset.categories) // classifier_config.batch_size()
        return self.num_batches

    def __getitem__(self, batch_id):
        max_points_num = dataset.find_max_points_num()  # max([len(sketch) for sketch in dataset.__train_data_list__[batch_id: batch_id + classifier_config.batch_size()]])

        with tf.device("/gpu:1"):
            train_data_np = np.zeros(shape=(classifier_config.batch_size(), max_points_num, 3))
            for i in range(classifier_config.batch_size()):
                if (batch_id + i) >= len(dataset.__train_data_list__):
                    break
                sketch_np = dataset.__train_data_list__[batch_id + i]
                points_num = len(sketch_np)
                train_data_np[i, 0:points_num] = sketch_np


            train_label_np = to_categorical(dataset.__train_label_list__[batch_id:
                                                                              batch_id + classifier_config.batch_size()],
                                            num_classes=len(dataset.categories))

        return train_data_np, train_label_np

    # def on_epoch_end(self):
    #     return


class TestDataSequence(Sequence):
    """
    Test data sequence generator.
    -> Give shuffled test data sequences.
    Note: `Train Data Sequence` and `Test Data Sequence` cannot be used on the same run.
    """

    def __init__(self):
        self.num_batches = -1
        # # self.on_epoch_end()
        # global epoch_shuffle_seeds, epoch_shuffle_seeds_idx
        # seed = epoch_shuffle_seeds[epoch_shuffle_seeds_idx]
        # epoch_shuffle_seeds_idx = (epoch_shuffle_seeds_idx + 1) % len(epoch_shuffle_seeds)

        # dataset.shuffle_test_data(seed)
        return

    def __len__(self):
        if self.num_batches == -1:
            self.num_batches = (classifier_config.num_per_category() - classifier_config.num_training_per_category()) * \
                               len(dataset.categories) // classifier_config.batch_size()
        return self.num_batches

    def __getitem__(self, batch_id):
        max_points_num = dataset.find_max_points_num()   # max([len(sketch) for sketch in dataset.__train_data_list__[batch_id: batch_id + classifier_config.batch_size()]])

        test_data_np = np.zeros(shape=(classifier_config.batch_size(), max_points_num, 3))
        for i in range(classifier_config.batch_size()):
            if (batch_id + i) >= len(dataset.__test_data_list__):
                break
            sketch_np = dataset.__test_data_list__[batch_id + i]
            points_num = len(sketch_np)
            test_data_np[i, 0:points_num] = sketch_np

        # test_data_np = np.asarray(test_data_np)

        test_label_np = to_categorical(dataset.__test_label_list__[batch_id:
                                                                        batch_id + classifier_config.batch_size()],
                                       num_classes=len(dataset.categories))

        if len(test_label_np) != classifier_config.batch_size():
            raise AssertionError(Fore.RED + "len(test_label_np) != classifier_config.batch_size()",
                                 "[batch_id: batch_id + classifier_config.batch_size()] == [{}, {}]".format(batch_id,
                                                                                                            batch_id + classifier_config.batch_size()))

        # test_label_np = np.asarray(test_label_np)
        return test_data_np, test_label_np


def main():
    global __EVALUATION_MODE__, epoch_shuffle_seeds_idx, __TRAIN_AND_EVALUATE_

    parser = ArgumentParser()
    parser.add_argument("-e", "--evaluate", help="evaluate model", default=False, action='store_true')
    parser.add_argument("-te", "--train_evaluate", help="train and immediately evaluate model afterwards",
                        default=False, action='store_true')
    parser.add_argument("-r", "--reduced", help="reduced the loaded number of data", type=float, default=1.)
    # parser.add_argument("-g", "--gpu", help="train on gpu", type=int, default=1)
    parser.add_argument("-d", "--debug", help="debug program", default=False, action='store_true')

    parser.add_argument("-s", "--starting_epoch",  # now unused
                        help="specify the starting epoch of the training, typically used when resuming training",
                        type=int, default=1)
    args = vars(parser.parse_args())

    if args["evaluate"] is True:
        print(Fore.GREEN + "In evaluation mode.")
        __EVALUATION_MODE__ = True
    if args["reduced"] > 1.:
        print(Fore.GREEN + "Reduced data to be loaded by {}.".format(args["reduced"]))
        classifier_config.reduce_loaded_data(args["reduced"])
    if args["train_evaluate"] is True:
        print("Train and evaluate model.")
        __TRAIN_AND_EVALUATE_ = True

    starting_epoch: int = args["starting_epoch"]
    if 1 < starting_epoch <= classifier_config.epoch():
        # note: epoch_shuffle_seeds_idx is inited to be 0,
        #   here, `starting_epoch_idx - 1 > 0` suffices
        epoch_shuffle_seeds_idx = starting_epoch - 1
        print(Fore.GREEN + "start(resume) training from epoch", starting_epoch)

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu"])
    # print("running on gpu", args["gpu"])

    dataset.load(shuffle_each_category=False, verbose=True,
                      num_per_category=classifier_config.num_per_category(),
                      num_training_per_category=classifier_config.num_training_per_category(),
                      test_only=__EVALUATION_MODE__,
                      load_cat_only=args["debug"])

    csv_logger = CSVLogger(classifier_config.csv_filepath(), append=True, separator=';')

    def do_evaluation():
        dataset.shuffle_test_data(epoch_shuffle_seeds[0])

        current_epoch: int = starting_epoch - 1
        while current_epoch < classifier_config.epoch():
            model_path: str = classifier_config.classifier_model_path() \
                              + "epoch_{}/".format(current_epoch + 1)

            global classifier_model
            classifier_model = models.load_model(model_path)

            print(classifier_model.summary())

            print("=" * 50)
            print("\tepoch {}".format(current_epoch + 1))
            print("=" * 50)
            result = classifier_model.evaluate(x=TestDataSequence(), use_multiprocessing=True, workers=4)
            print(result)

            current_epoch = current_epoch + 1
        return

    def do_training():
        global classifier_model
        if starting_epoch == 1:
            hidden_size = classifier_config.lstm_hidden_size()
            classifier_model = models.Sequential()
            classifier_model.add(layers.Masking(mask_value=0.,
                                                input_shape=(dataset.find_max_points_num(), 3)))
            classifier_model.add(layers.Dropout(0.2))
            classifier_model.add(layers.LSTM(
                units=hidden_size,
                return_sequences=True,

            ))
            classifier_model.add(layers.LSTM(units=hidden_size, return_sequences=True))
            classifier_model.add(layers.LSTM(units=hidden_size, return_sequences=False))
            classifier_model.add(layers.Dropout(0.2))
            classifier_model.add(layers.Dense(units=len(dataset.categories), activation='softmax'))
            classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model_path: str = classifier_config.classifier_model_path() \
                              + "epoch_{}/".format(starting_epoch - 1)
            classifier_model = models.load_model(model_path)
            print("models loaded from", model_path)

        print(classifier_model.summary())

        classifier_model.fit(x=TrainDataSequence(),
                             epochs=(classifier_config.epoch() - starting_epoch + 1),
                             # `starting_epoch` starts from 1
                             callbacks=[
                                 # cp_callback,
                                 csv_logger,
                                 OnEpochEnd()
                             ],
                             use_multiprocessing=True, workers=8
                             )
        return


    if __EVALUATION_MODE__ is True:
        do_evaluation()
    else:
        do_training()
        if __TRAIN_AND_EVALUATE_ is True:
            do_evaluation()
    return


if __name__ == '__main__':
    print(epoch_shuffle_seeds)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()
