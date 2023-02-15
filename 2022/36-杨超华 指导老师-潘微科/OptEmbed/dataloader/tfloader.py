import tensorflow as tf
import glob
import torch
import os

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
        files = glob.glob(repo_path + '/' + self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
                        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x,y in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
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
        files = glob.glob(repo_path + '/' + self.tfrecord_path + '/' + "{}*".format(data_type))
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
        files = glob.glob(repo_path + '/' + self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
                        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x,y in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            yield x, y