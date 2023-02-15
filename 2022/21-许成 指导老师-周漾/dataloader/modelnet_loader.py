import os
import numpy as np
import warnings
import pickle
from Generator.config import opts
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

class ModelNetDataLoader(Dataset):
    def __init__(self,root,args,split = 'train',process_data = False):
        self.root = root
        self.npoint = opts.np
        self.process_data = process_data

        self.catfile = os.path.join(self.root,'data/ModelNet40')

        self.cat = [line.rstrip() for line in open(self.catfile)]

        self.classes = dict(self.cat,range(len(self.cat)))

       