import numpy as np
import os
import h5py

class Coreset(object):
    dataset = None
    data = None
    coreset = None
    coreset_label = None
    label = None

    # init the path of dataset
    def __init__(self, dataset = None):
        if dataset is not None:
            print("Init with dataset name : ", dataset)
        else:
            print("Dataset is None, please define it before loading.")
        # path to the file of dataset
        self.dataset = dataset
        # data
        self.data = np.array([])
        # label used to calculate scores
        self.label = None
        # sampled coreset
        self.coreset = None
        # coreset weight
        self.weight = None
        # coreset label used to calculate scores
        self.coreset_label = None

    def define_dataset(self, dataset):
        print("Define path to dataset : {}".format(dataset))
        self.dataset = dataset

    def get_coreset(self):
        # if coreset is not define return full data
        if self.coreset is None:
            print('WARNING: Coreset is not defined. Please check your coreset construction. Return full dataset.')
            return self.data
        return self.coreset

    def load_data(self, data = None, label = None):
        print("Load data from outer data")
        self.data = data
        self.label = label
        return self


    # f : user defined function to load data
    def load_UDF(self, f):
        print("Load data from UDF")
        # load data from dataset
        self.data = f(self.dataset)
        return self

    # size : the size of coreset
    # coreset_config : {'cluster' : cluster number, 'delta' : probability of coreset, 'epsilon' : error bounding}
    def construct(self, size = 0, **coreset_config):
        print("-------------------Constructing Coreset-------------------")
        # the size of rows
        m = self.data.shape[0]
        print("Row size of data:", m)
        # the size of columns
        n = self.data.shape[1]
        print("Dimensions(Features) of data:", n)

        # Coreset size computed by hyper parameters
        if size == 0:
            if coreset_config.get("cluster") is None or coreset_config.get("delta") is None or coreset_config.get("epsilon") is None:
                raise Exception("Lacking hyper parameters of coreset")
            size = np.round((n*coreset_config['cluster']*np.log(coreset_config['cluster'])+np.log(1/coreset_config['delta']))/(coreset_config['epsilon']**2)).astype(np.int64)
            print("size of coreset={} calculated by dim={} cluster={} delta={} epsilon={}".format(size, n, coreset_config['cluster'], coreset_config['delta'], coreset_config['epsilon']))
        else:
            print("size of coreset={} decide by user".format(size))

        # store the sum of rows
        u = np.zeros(n, dtype=np.float)
        for row in self.data:
            u += row
        # average -> mean
        u *= 1.0 / m
        print("Average all rows : ", u)

        # distance of each data point to mean
        # q[i] = \sqrt{\sum ||x-u||^2} ^2 【quantization error】
        q = np.zeros(m, dtype=np.float)

        # sum of all distance
        sum = 0
        for i in range(0, m):
            s = 0
            # \sum ||x-u||^2
            for j in range(0, n):
                s += self.data[i][j]**2 + u[j]**2 - 2*self.data[i][j]*u[j]
            q[i] = s
            sum += s
        print("Sum all distance : ", sum)

        # get distribution function
        for i in range(0, m):
            q[i] = 0.5 * (q[i] / sum + 1.0 / m)
        print("Compute distribution of all data points")

        # distribution sampling : draw ‘size' from 'm' data
        # the index of samples
        samplei = np.random.choice(m, size, p=q)
        print("Choose points from dataset")

        sample = np.zeros((0, n), dtype=np.float)
        sample_label = np.zeros(size, dtype=np.int64)
        weight = np.zeros(size, dtype=np.float)
        print("Start sampling from dataset")
        if self.label is not None:
            print("Samples with labels")
            # load labels from dataset
            for i in range(0, size):
                sample = np.row_stack((sample, self.data[samplei[i]]))
                sample_label[i] = self.label[samplei[i]]
                weight[i] = 1. / (size * q[samplei[i]])
        else:
            print("Samples without labels")
            # not load labels from dataset
            for i in range(0, size):
                sample = np.row_stack((sample, self.data[samplei[i]]))
                weight[i] = 1. / (size * q[samplei[i]])
        print("End sampling")

        print("-------------------Construction Finished-------------------")

        self.coreset = sample
        self.coreset_label = sample_label
        self.weight = weight

    def writeH5(self):
        print("Creating H5 File")
        if not os.path.exists("./h5/"):
            print("h5 dir not exist, try to mkdir")
            os.mkdir("./h5/")
            print("mkdir complete")

        f = h5py.File(r"./h5/coreset-" + self.dataset, "w")
        print("Writing data into dataset['train/data']")
        dataset = f.create_dataset("train/data", (self.coreset.shape[0], self.coreset.shape[1]), dtype="i8", data=self.coreset)
        print("Writing label into dataset['train/label']")
        labelset = f.create_dataset("train/label", self.coreset_label.size, dtype="i8", data=self.coreset_label)
        return (dataset, labelset)

