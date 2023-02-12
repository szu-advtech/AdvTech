from transform import DataTransform
from datetime import datetime, date
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description='Transfrom original data to TFRecord')

#parser.add_argument('avazu', type=string)
# parser.add_argument('--label', type=str)
# parser.add_argument("--store_stat", action="store_true")
# parser.add_argument("--threshold", type=int, default=0)
# parser.add_argument("--dataset", type=Path)
# parser.add_argument("--record", type=Path)
# parser.add_argument("--ratio", nargs='+', type=float)
parser.add_argument('--label', type=str, default="Label")
parser.add_argument("--store_stat", action="store_true", default=True)
parser.add_argument("--threshold", type=int, default=10)
parser.add_argument("--dataset", type=Path, default='../../datasets/original/kdd12/training.txt')
parser.add_argument("--stats", type=Path, default='../data/kdd12/stats')
parser.add_argument("--record", type=Path, default='../data/kdd12/tfrecord')
parser.add_argument("--ratio", nargs='+', type=float, default=[0.8, 0.1, 0.1])
args = parser.parse_args()

class KDDTransform(DataTransform):
    def __init__(self, dataset_path, path, stats_path, min_threshold, label_index, ratio, store_stat = False, seed = 2021):
       super(KDDTransform, self).__init__(dataset_path, stats_path, store_stat=store_stat, seed=seed )
       self.threshold = min_threshold
       self.label = label_index
       self.split = ratio
       self.path = path
       self.stats_path = stats_path
       self.name ="Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11".split(",")
    
    def process(self):
     
        self._read(name = self.name, header = None, label_index = self.label,sep="\t")
        if self.store_stat:
            self.generate_and_filter(threshold=self.threshold, label_index=self.label)
        tr, te, val = self.random_split(ratio=self.split)
        self.transform_tfrecord(tr, self.path, "train", label_index=self.label)
        self.transform_tfrecord(te, self.path, "test", label_index=self.label)
        self.transform_tfrecord(val, self.path, "validation", label_index=self.label)

    def _process_x(self):
        print(self.data[self.data["Label"] == 1].shape)
    
    def _process_y(self):
        self.data["Label"] = self.data["Label"].apply(lambda x: 0 if x == 0 else 1)

if __name__ == "__main__":
    tranformer = KDDTransform(args.dataset, args.record, args.stats,
                                 args.threshold, args.label,
                                 args.ratio, store_stat=args.store_stat)
    tranformer.process()
