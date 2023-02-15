import numpy as np
import pickle
import os
import copy
import torch

from torchvision import datasets, transforms
from torch.utils import data
from tool.parameter import args_parser


def get_dataset1():
    args=args_parser()
    trans_mnist = transforms.Compose([transforms.ToTensor() ,transforms.Normalize((0.1307) ,(0.3081))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

    if args.iid:
        num_items = int(len(dataset_train) / args.num_users)
        dict_users = {}
        idxs = [i for i in range(len(dataset_train))]
        for i in range(args.num_users):
            dict_users[i] = set(np.random.choice(idxs, num_items, replace=False))
            idxs = list(set(idxs) - dict_users[i])
    else:
        dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
        idxs = np.arange(len(dataset_train))
        labels = dataset_train.train_labels.numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels)) # 竖直拼接两个数组
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # argsort()函数是对数组中的元素进行从小到大排序，并返回相应序列元素的数组下标
        idxs = idxs_labels[0, :]

        m = int(args.num_users/2)
        n = args.num_users
        a = 196
        lens = int(len(dataset_train)/10)
        # divide and assign
        for i in range(m):
            idx=[]
            for j in range(5):
                idx_t = idxs[j*lens:(j+1)*lens]
                temp=list(np.random.choice(idx_t, a, replace=False))
                idx+=temp
            dict_users[i] = idx

        j=0
        for i in range(m,n):
            idx = []
            idx_t = idxs[(j) * lens:(j+1) * lens]
            idx_t1 = idxs[(j+5)* lens:(j+6) * lens]
            j=(j+1)%5
            idx+=list(np.random.choice(idx_t, int(a/2), replace=False))+list(
                np.random.choice(idx_t1, 2*a, replace=False))
            dict_users[i] = idx

    return dataset_train,dataset_test,dict_users

class CifarDataset(data.Dataset):
    def __init__(self, xs, ys, is_train=True):
        self.xs = copy.deepcopy(xs)
        self.ys = copy.deepcopy(ys)

        if is_train is True:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        img = self.xs[index]
        label = self.ys[index]

        img = img.transpose((1, 2, 0)).astype(np.uint8)
        img = self.transform(img)

        img = torch.FloatTensor(img)
        label = torch.LongTensor([label])[0]
        return img, label

def get_dataset2():
    args = args_parser()

    cifar_fdir = "../data/Cifar/CIFAR10/CIFAR10"
    cifar_fpaths = {
        "cifar10": {
            "train_fpaths":[
                os.path.join(cifar_fdir, "cifar10-train-part1.pkl"),
                os.path.join(cifar_fdir, "cifar10-train-part2.pkl"),
            ],

            "test_fpath": os.path.join(cifar_fdir, "cifar10-test.pkl")
        },
    }

    train_xs, train_ys = [],[]
    cifar_fpaths1 = cifar_fpaths["cifar10"]["train_fpaths"]
    for fpath in cifar_fpaths1:
        with open(fpath, "rb") as fr:
            data_train = pickle.load(fr)
        train_xs.append(data_train["images"])
        train_ys.append(data_train["labels"])
    train_xs = np.concatenate(train_xs, axis=0)
    train_ys = np.concatenate(train_ys, axis=0)

    idxs = np.arange(len(train_xs))
    idxs_labels = np.vstack((idxs, train_ys))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    cifar_fpaths2 = cifar_fpaths["cifar10"]["test_fpath"]
    with open(cifar_fpaths2, "rb") as fr:
        data_test = pickle.load(fr)
    test_xs = data_test["images"]
    test_ys = data_test["labels"]

    dataset_train=CifarDataset(train_xs,train_ys)
    dataset_test=CifarDataset(test_xs,test_ys,False)
    #trans_cifar = transforms.Compose([transforms.ToTensor() ,transforms.Normalize((0.1307) ,(0.3081))])

    dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
    m = int(args.num_users / 2)
    n = args.num_users
    a = 68
    lens = int(len(dataset_train) / 10)
    # divide and assign
    for i in range(m):
        idx = []
        for j in range(5):
            idx_t = idxs[j * lens:(j + 1) * lens]
            temp = list(np.random.choice(idx_t, a, replace=False))
            idx = idx + temp
        dict_users[i] = idx

    j=0
    for i in range(m, n):
        idx = []
        idx_t = idxs[(j) * lens:(j + 1) * lens]
        idx_t1 = idxs[(j + 5) * lens:(j + 6) * lens]
        j = (j + 1) % 5
        idx += list(np.random.choice(idx_t, int(a / 2), replace=False)) + list(
            np.random.choice(idx_t1, 2 * a, replace=False))
        dict_users[i] = idx
    return dataset_train,dataset_test,dict_users