import tensorflow as tf
import glob
import torch
import os
import copy

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CriteoLoader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 39
        self.tfrecord_path = tfrecord_path
        self.description = {
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
        }
    
    def get_data(self, data_type, batch_size = 1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label']
        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        #print(files)
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
                        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x,y in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            print(x)
            print(x.shape)
            print(y)
            print(y.shape)

            break
            yield x, y

class Avazuloader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 24
        self.tfrecord_path = tfrecord_path
        self.description = {
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
        }

    def get_data(self, data_type, batch_size = 1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label']
        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
                        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x,y in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            yield x, y

class KDD12loader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 11
        self.tfrecord_path = tfrecord_path
        self.description = {
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
        }

    def get_data(self, data_type, batch_size = 1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label']
        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
                        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x,y in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            yield x, y


class AliExpressLoader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 79
        self.tfrecord_path = tfrecord_path
        self.data = None
        self.description = {
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "domain": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
        }

    def get_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label'], example['domain']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        # print(files)
        data = []
        if data_type == 'train':
            if self.data == None:
                ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
                    batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

                # ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
                #     batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(buffer_size=400, seed=2022, reshuffle_each_iteration=False)
                self.data = ds
            else:
                ds = self.data
        else:
            ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
                batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x, y, d in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            d = torch.from_numpy(d.numpy())
            # print(x)
            # print(x.shape)
            # print(torch.sum(y))
            # print(y.shape)
            #print(torch.sum(d))
            #print(d.shape)
            #
            # break
            #data.append([x, y, d])
            yield x, y, d
        #return data
