from transform import DataTransform
from datetime import datetime, date
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser(description='Transfrom original data to TFRecord')

#parser.add_argument('avazu', type=string)

parser.add_argument('--label', type=str, default="Label")
parser.add_argument("--store_stat", action="store_true", default=True)
parser.add_argument("--threshold", type=int, default=2)
parser.add_argument("--dataset1", type=Path, default='../data/nl/nl_train.csv')
parser.add_argument("--dataset2", type=Path, default='../data/nl/nl_test.csv')
parser.add_argument("--stats", type=Path, default='../data/nl/stats_2_4')
parser.add_argument("--record", type=Path, default='../data/nl/threshold_2_4')
parser.add_argument("--ratio", nargs='+', type=float, default=[0.7, 0.15, 0.15])

args = parser.parse_args()


class AilExpressTransform(DataTransform):
    def __init__(self, dataset_path1, dataset_path2, path, stats_path, min_threshold, label_index, ratio, store_stat=False, seed=2021):
        super(AilExpressTransform, self).__init__(dataset_path1, dataset_path2, stats_path, store_stat=store_stat, seed=seed)
        self.threshold = min_threshold
        self.label = label_index
        self.split = ratio
        self.path = path
        self.stats_path = stats_path
        self.name = "Id,U1,U2,U3,U4,U5,U6,U7,U8,U9,U10,U11,U12,U13,U14,U15,U16,U17,U18,U19,U20,U21,U22,U23,U24,U25,U26,U27,U28,U29,U30,U31,U32,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I16,I17,I18,I19,I20,I21,I22,I23,I24,I25,I26,I27,I28,I29,I30,I31,I32,I33,I34,I35,I36,I37,I38,I39,I40,I41,I42,I43,I44,I45,I46,I47,Label".split(
            ",")

    def process(self):
        self._read(name=self.name, header=None, sep=",", label_index=self.label)
        if self.store_stat:
            self.generate_and_filter(threshold=self.threshold, label_index=self.label)
        tr, te, val = self.random_split(ratio=self.split)
        self.transform_tfrecord(tr, self.path, "train", label_index=self.label)
        self.transform_tfrecord(te, self.path, "test", label_index=self.label)
        self.transform_tfrecord(val, self.path, "validation", label_index=self.label)

    def _process_x(self):
        print(self.data[self.data["Label"] == 1].shape)
        # print(self.train_data[self.train_data["Label"] == 1].shape)
        # print(self.test_data[self.test_data["Label"] == 1].shape)

    def _process_y(self):
        self.data = self.data.drop("Id", axis=1)
        # self.train_data = self.train_data.drop("Id", axis=1)
        # self.test_data = self.test_data.drop("Id", axis=1)
        self.data["Label"] = self.data["Label"].apply(lambda x: 0 if x == 0 else 1)
        # self.train_data["Label"] = self.train_data["Label"].apply(lambda x: 0 if x == 0 else 1)
        # self.test_data["Label"] = self.test_data["Label"].apply(lambda x: 0 if x == 0 else 1)

if __name__ == "__main__":
    tranformer = AilExpressTransform(args.dataset1, args.dataset2, args.record, args.stats,
                                 args.threshold, args.label,
                                 args.ratio, store_stat=args.store_stat)
    tranformer.process()
