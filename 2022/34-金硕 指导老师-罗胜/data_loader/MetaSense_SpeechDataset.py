import numpy as np

import torch.utils.data
import pandas as pd
import time
import sys

import multiprocessing as mp
manager = mp.Manager()

sys.path.append('..')
import options

opt = options.MetaSense_SpeechOpt
WIN_LEN = opt['seq_len']

class KSHOTTensorDataset(torch.utils.data.Dataset):
    def __init__(self, num_classes, features, classes, domains):
        assert (features.shape[0] == classes.shape[0] == domains.shape[0])

        self.num_classes = num_classes
        self.features_per_class = []
        self.classes_per_class = []
        self.domains_per_class = []

        for class_idx in range(self.num_classes):
            indices = np.where(classes == class_idx)
            self.features_per_class.append(np.random.permutation(features[indices]))
            self.classes_per_class.append(np.random.permutation(classes[indices]))
            self.domains_per_class.append(np.random.permutation(domains[indices]))

        self.data_num = min(
            [len(feature_per_class) for feature_per_class in self.features_per_class])  # get min number of classes

        for i in range(self.num_classes):
            self.features_per_class[i] = torch.from_numpy(self.features_per_class[i][:self.data_num]).float()
            self.classes_per_class[i] = torch.from_numpy(self.classes_per_class[i][:self.data_num])
            self.domains_per_class[i] = torch.from_numpy(self.domains_per_class[i][:self.data_num])

    def __getitem__(self, index):

        features = torch.FloatTensor(self.num_classes, *(self.features_per_class[0][0].shape)) # make FloatTensor with shape num_classes x F-dim1 x F-dim2...
        classes = torch.LongTensor(self.num_classes)
        domains = torch.LongTensor(self.num_classes)

        rand_indices = [i for i in range(self.num_classes)]
        np.random.shuffle(rand_indices)

        for i in range(self.num_classes):
            features[i] = self.features_per_class[rand_indices[i]][index]
            classes[i] = self.classes_per_class[rand_indices[i]][index]
            domains[i] = self.domains_per_class[rand_indices[i]][index]

        return (features, classes, domains)

    def __len__(self):
        return self.data_num


class MetaSense_SpeechDataset(torch.utils.data.Dataset):
    # load static files

    def __init__(self, file='../dataset/metasense_speech/minmax_scaling_all.csv', transform=None, domain=None, word=None, complementary=False,
                 max_source=100):
        """
        Args:
            file_path (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            domain: condition on user-phone combination
            word: word
            complementary: is it complementary dataset for given conditions? (used for "multi" case)
            
        """
        st = time.time()
        self.domain = domain
        self.word = word
        self.complementary = complementary
        self.max_source=max_source

        self.df = pd.read_csv(file)
        if complementary:  # for multi domain
            if domain:
                self.df = self.df[self.df['domain'] != domain]
            if word:
                self.df = self.df[self.df['word'] != word]
        else:
            if domain:
                self.df = self.df[self.df['domain'] == domain]
            if word:
                self.df = self.df[self.df['word'] == word]
            # print(len(self.df))

        self.transform = transform
        ppt = time.time()

        self.preprocessing()
        print('Loading data done with rows:{:d}\tPreprocessing:{:f}\tTotal Time:{:f}'.format(len(self.df.index),
                                                                                             time.time() - ppt,
                                                                                             time.time() - st))
    def preprocessing(self):
        self.num_domains = 0
        self.features = []
        self.class_labels = []
        self.domain_labels = []

        self.datasets = []  # list of dataset per each domain
        self.kshot_datasets = []  # list of dataset per each domain

        if self.complementary:
            domains = set(options.MetaSense_SpeechOpt['domains']) - set(self.domain)  # currently supports only user and position
        else:
            domains = set([self.domain])  # bracket is required to append a tuple
        domains = list(domains)
        # domain_superset = list(itertools.product(positions, users))
        valid_domains = []

        for idx in range(max(len(self.df) // WIN_LEN, 0)):
            domain = self.df.iloc[idx * WIN_LEN, 2]
            class_label = self.df.iloc[idx * WIN_LEN, 1]
            domain_label = -1

            for i in range(len(domains)):
                if domains[i] == domain and domains[i] not in valid_domains:
                    valid_domains.append(domains[i])
                    break

            if domain in valid_domains:
                domain_label = valid_domains.index(domain)
            else:
                continue
            feature = self.df.iloc[idx * WIN_LEN:(idx + 1) * WIN_LEN, 0:1].values
            feature = feature.T

            self.features.append(feature)
            self.class_labels.append(self.class_to_number(class_label))
            self.domain_labels.append(domain_label)

        self.num_domains = len(valid_domains) if len(valid_domains) < self.max_source else self.max_source
        print('Valid domains:' + str(valid_domains[:self.num_domains]))
        self.features = np.array(self.features, dtype=np.float)
        self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domain_labels)

        # append dataset for each domain
        for domain_idx in range(self.num_domains):
            indices = np.where(self.domain_labels == domain_idx)[0]
            self.datasets.append(torch.utils.data.TensorDataset(torch.from_numpy(self.features[indices]).float(),
                                                                torch.from_numpy(self.class_labels[indices]),
                                                                torch.from_numpy(self.domain_labels[indices])))
            kshot_dataset = KSHOTTensorDataset(len(np.unique(self.class_labels)),
                                                          self.features[indices],
                                                          self.class_labels[indices],
                                                          self.domain_labels[indices])
            self.kshot_datasets.append(kshot_dataset)

        # concated dataset
        self.dataset = torch.utils.data.ConcatDataset(self.datasets)

    def __len__(self):
        # return max(len(self.df) // OVERLAPPING_WIN_LEN - 1, 0)
        return len(self.dataset)

    def get_num_domains(self):
        return self.num_domains

    def get_datasets_per_domain(self):
        return self.kshot_datasets

    def class_to_number(self, label):
        dic = {'yes': 0,
                'no': 1,
                'up': 2,
                'down': 3,
                'left': 4,
                'right': 5,
                'on': 6,
                'off': 7,
                'stop': 8,
                'go': 9,
                'forward': 10,
                'backward': 11,
                'follow': 12,
                'learn': 13}
        return dic[label]


    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        return self.dataset[idx]
    



if __name__ == "__main__":
    pass