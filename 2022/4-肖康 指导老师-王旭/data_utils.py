from os import listdir
from os.path import join
import numpy as np
from PIL import Image
import scipy.io as scio
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch
import ipdb

def train_hr_transform(crop_size):

    return Compose([RandomCrop(crop_size),ToTensor(),])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size,patches, istrain=True, data_path= './imagedata/DMOS_SIQAD.mat', si=49, sj=7, indexlist=(10,22)):
        super(TrainDatasetFromFolder, self).__init__()
        datafilenames1=[]
        datafilenames2 = []
        datafilenames3 = []
        datafilenames4 = []
        train_validation_label1 = []
        train_noise_strengh1 = []
        data = scio.loadmat(data_path)
        load_matrix = data['DMOS']
        self.istrain = istrain
        for i in range(20):
            for j in range(7):
                for k in range(7):
                    if (i+1) in indexlist:
                        for m in range(patches):
                            datafilenames1.append(dataset_dir+ '/cim'+str(i+1)+'_'+str(j+1)+'_'+str(k+1)+'_1.bmp')
                            datafilenames2.append(
                                dataset_dir + '/cim' + str(i + 1) + '_' + str(j + 1) + '_' + str(k + 1) + '_2.bmp')
                            datafilenames3.append(
                                dataset_dir + '/cim' + str(i + 1) + '_' + str(j + 1) + '_' + str(k + 1) + '_3.bmp')
                            datafilenames4.append(
                                dataset_dir + '/cim' + str(i + 1) + '_' + str(j + 1) + '_' + str(k + 1) + '_4.bmp')
                            train_noise_strengh1.append(7*j+k)
                            train_validation_label1.append(np.float32(load_matrix[i*si+sj*j+k]))

        temp = np.array([datafilenames1,datafilenames2,datafilenames3,datafilenames4, train_validation_label1])
        temp = temp.transpose()
        np.random.shuffle(temp)
        datafilenames1_1 = list(temp[:, 0])
        datafilenames1_2 = list(temp[:, 1])
        datafilenames1_3 = list(temp[:, 2])
        datafilenames1_4 = list(temp[:, 3])
        train_validation_label2= list(temp[:, 4])
        train_validation_label2 = [np.float32(i) for i in train_validation_label2]

        self.image_filenames1=datafilenames1
        self.image_filenames2 = datafilenames2
        self.image_filenames3 = datafilenames3
        self.image_filenames4 = datafilenames4
        self.hr_transform = train_hr_transform(crop_size)
        self.type=train_noise_strengh1
        self.label1=train_validation_label1

        self.image_filenames1_1=datafilenames1_1
        self.image_filenames2_1 = datafilenames1_2
        self.image_filenames3_1 = datafilenames1_3
        self.image_filenames4_1 = datafilenames1_4
        self.label2=train_validation_label2

    def __getitem__(self, index):
        h1_image = self.hr_transform(Image.open(self.image_filenames1[index]))
        h2_image = self.hr_transform(Image.open(self.image_filenames2[index]))
        h3_image = self.hr_transform(Image.open(self.image_filenames3[index]))
        h4_image = self.hr_transform(Image.open(self.image_filenames4[index]))
        type=self.type[index]
        label1=self.label1[index]
        if self.istrain:
            h5_image = self.hr_transform(Image.open(self.image_filenames1_1[index]))
            h6_image = self.hr_transform(Image.open(self.image_filenames2_1[index]))
            h7_image = self.hr_transform(Image.open(self.image_filenames3_1[index]))
            h8_image = self.hr_transform(Image.open(self.image_filenames4_1[index]))
            label2=self.label2[index]
            return  h1_image,h2_image,h3_image,h4_image,h5_image,h6_image,h7_image,h8_image,label1,type,label2
        else:
            return h1_image,h2_image,h3_image,h4_image,label1,type

    def __len__(self):
        return len(self.image_filenames1)