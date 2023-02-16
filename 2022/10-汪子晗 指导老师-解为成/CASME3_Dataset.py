import numpy as np
from torchvision import transforms
import torch
import torch.utils.data as data
import os
import re
import cv2

def make_dataset(onset,apex,emo):
    return [(onset[i], apex[i], emo[i]) for i in range(len(onset))]

class CASME3_4(data.Dataset):
    def __init__(self,root_path,train,num_loso,transform=None,basic_aug=False,transform_norm=None):
        self.root_path = root_path
        self.train = train
        self.num_loso = num_loso
        self.transform = transform
        self.transform_norm = transform_norm
        self.basic_aug = basic_aug
        self.subject_out_idx = num_loso
        self.img_path = open(os.path.join('/NAS_REMOTE/wangzihan', 'list6/CASME_onset_img_path.txt')).readlines()
        self.idx = []

        for i in range(len(self.img_path)):
            num = self.img_path[i].split('/')[0]
            num = int(re.sub('spNO.', '', num))
            if num == self.subject_out_idx:
                self.idx.append(i)
        if len(self.idx):
            start = self.idx[0]
            end = self.idx[-1]
        else:
            end,start = -1,0
        self.onset = open(os.path.join(self.root_path, 'list6', 'CASME_onset_img_path.txt')).readlines()
        self.apex = open(os.path.join(self.root_path, 'list6', 'CASME_apex_img_path.txt')).readlines()
        self.emotion = np.loadtxt(os.path.join(self.root_path, 'list6', 'CASME_emo_label.txt'))

        if self.train:
            train_onset = self.onset[:start] + self.onset[end+1:]
            train_apex = self.apex[:start] + self.apex[end+1:]
            train_emo = np.delete(self.emotion, self.idx, 0)
            self.data_list = make_dataset(train_onset, train_apex, train_emo)
        else:
            test_onset = self.onset[start:end+1]
            test_apex = self.apex[start:end+1]
            test_emo = self.emotion[self.idx, :]
            self.data_list = make_dataset(test_onset, test_apex, test_emo)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        onset, apex, emo = self.data_list[item]
        onset = '/data2/wangzihan/part_A_cropped/' + onset.strip()
        apex = '/data2/wangzihan/part_A_cropped/' + apex.strip()
        img_onset = cv2.imread(onset)
        img_apex = cv2.imread(apex)

        if self.transform is not None:
            img_onset = self.transform(img_onset)
            img_apex = self.transform(img_apex)
            all = torch.cat((img_onset, img_apex), dim=0)
            if self.transform_norm is not None and self.train:
                all = self.transform_norm(all)
            img_onset = all[0:3,:,:]
            img_apex = all[3:6,:,:]
        emo = np.argmax(emo)

        return img_onset, img_apex, emo


class CASME3_7(data.Dataset):
    def __init__(self,root_path,train,num_loso,transform=None,basic_aug=False,transform_norm=None):
        self.root_path = root_path
        self.train = train
        self.num_loso = num_loso
        self.transform = transform
        self.transform_norm = transform_norm
        self.basic_aug = basic_aug
        self.subject_out_idx = num_loso
        self.img_path = open(os.path.join('/NAS_REMOTE/wangzihan', 'list6/CASME_onset_img_path.txt')).readlines()
        self.idx = []

        for i in range(len(self.img_path)):
            num = self.img_path[i].split('/')[0]
            num = int(re.sub('spNO.', '', num))
            if num == self.subject_out_idx:
                self.idx.append(i)
        if len(self.idx):
            start = self.idx[0]
            end = self.idx[-1]
        else:
            end,start = -1,0
        self.onset = open(os.path.join(self.root_path, 'list6', 'CASME_onset_img_path.txt')).readlines()
        self.apex = open(os.path.join(self.root_path, 'list6', 'CASME_apex_img_path.txt')).readlines()
        self.emotion = np.loadtxt(os.path.join(self.root_path, 'list6', 'CASME_emo_7classes_label.txt'))

        if self.train:
            train_onset = self.onset[:start] + self.onset[end+1:]
            train_apex = self.apex[:start] + self.apex[end+1:]
            train_emo = np.delete(self.emotion, self.idx, 0)
            self.data_list = make_dataset(train_onset, train_apex, train_emo)
        else:
            test_onset = self.onset[start:end+1]
            test_apex = self.apex[start:end+1]
            test_emo = self.emotion[self.idx, :]
            self.data_list = make_dataset(test_onset, test_apex, test_emo)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        onset, apex, emo = self.data_list[item]
        onset = '/data2/wangzihan/part_A_cropped/' + onset.strip()
        apex = '/data2/wangzihan/part_A_cropped/' + apex.strip()
        img_onset = cv2.imread(onset)
        img_apex = cv2.imread(apex)

        if self.transform is not None:
            img_onset = self.transform(img_onset)
            img_apex = self.transform(img_apex)
            all = torch.cat((img_onset, img_apex), dim=0)
            if self.transform_norm is not None and self.train:
                all = self.transform_norm(all)
            img_onset = all[0:3,:,:]
            img_apex = all[3:6,:,:]
        emo = np.argmax(emo)

        return img_onset, img_apex, emo

class CASME3_OF(data.Dataset):
    def __init__(self,root_path,train,num_loso,transform=None,basic_aug=False,transform_norm=None):
        self.root_path = root_path
        self.train = train
        self.num_loso = num_loso
        self.transform = transform
        self.transform_norm = transform_norm
        self.basic_aug = basic_aug
        self.subject_out_idx = num_loso
        self.img_path = open(os.path.join('/NAS_REMOTE/wangzihan', 'list6/CASME_onset_img_path.txt')).readlines()
        self.idx = []

        for i in range(len(self.img_path)):
            num = self.img_path[i].split('/')[0]
            num = int(re.sub('spNO.', '', num))
            if num == self.subject_out_idx:
                self.idx.append(i)
        if len(self.idx):
            start = self.idx[0]
            end = self.idx[-1]
        else:
            end,start = -1,0
        self.onset = open(os.path.join(self.root_path, 'list6', 'CASME_onset_img_path.txt')).readlines()
        self.apex = open(os.path.join(self.root_path, 'list6', 'CASME_apex_img_path.txt')).readlines()
        self.emotion = np.loadtxt(os.path.join(self.root_path, 'list6', 'CASME_emo_label.txt'))

        if self.train:
            train_onset = self.onset[:start] + self.onset[end+1:]
            train_apex = self.apex[:start] + self.apex[end+1:]
            train_emo = np.delete(self.emotion, self.idx, 0)
            self.data_list = make_dataset(train_onset, train_apex, train_emo)
        else:
            test_onset = self.onset[start:end+1]
            test_apex = self.apex[start:end+1]
            test_emo = self.emotion[self.idx, :]
            self.data_list = make_dataset(test_onset, test_apex, test_emo)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        onset, apex, emo = self.data_list[item]
        onset = '/data2/wangzihan/part_A_cropped/' + onset.strip()
        apex = '/data2/wangzihan/part_A_cropped/' + apex.strip()
        img_onset = cv2.imread(onset)
        img_onset = cv2.resize(img_onset,(224,224)).astype(np.float32)
        img_onset = img_onset/255.0
        img_onset = np.swapaxes(img_onset, 0, 2)
        img_onset = np.swapaxes(img_onset, 1, 2)
        #img_apex = cv2.imread(apex)
        flow_x = np.loadtxt(re.sub('.jpg', 'of_x.txt', apex.strip()))
        flow_y = np.loadtxt(re.sub('.jpg', 'of_y.txt', apex.strip()))
        flow_x = np.expand_dims(flow_x,axis=0)
        flow_y = np.expand_dims(flow_y,axis=0)
        flow = np.concatenate((flow_x, flow_y), axis=0).astype(np.float32)

        emo = np.argmax(emo)#.astype(np.float32)


        return img_onset, flow, emo

class CASME3_depth(data.Dataset):
    def __init__(self,root_path,train,num_loso,transform=None,basic_aug=False,transform_norm=None):
        self.root_path = root_path
        self.train = train
        self.num_loso = num_loso
        self.transform = transform
        self.transform_norm = transform_norm
        self.basic_aug = basic_aug
        self.subject_out_idx = num_loso
        self.img_path = open(os.path.join('/NAS_REMOTE/wangzihan', 'list6/CASME_onset_img_path.txt')).readlines()
        self.idx = []

        for i in range(len(self.img_path)):
            num = self.img_path[i].split('/')[0]
            num = int(re.sub('spNO.', '', num))
            if num == self.subject_out_idx:
                self.idx.append(i)
        if len(self.idx):
            start = self.idx[0]
            end = self.idx[-1]
        else:
            end,start = -1,0
        self.onset = open(os.path.join(self.root_path, 'list6', 'CASME_onset_img_path.txt')).readlines()
        self.apex = open(os.path.join(self.root_path, 'list6', 'CASME_apex_img_path.txt')).readlines()
        self.emotion = np.loadtxt(os.path.join(self.root_path, 'list6', 'CASME_emo_label.txt'))

        if self.train:
            train_onset = self.onset[:start] + self.onset[end+1:]
            train_apex = self.apex[:start] + self.apex[end+1:]
            train_emo = np.delete(self.emotion, self.idx, 0)
            self.data_list = make_dataset(train_onset, train_apex, train_emo)
        else:
            test_onset = self.onset[start:end+1]
            test_apex = self.apex[start:end+1]
            test_emo = self.emotion[self.idx, :]
            self.data_list = make_dataset(test_onset, test_apex, test_emo)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        onset, apex, emo = self.data_list[item]
        onset = '/data2/wangzihan/part_A_cropped/' + onset.strip()
        apex = '/data2/wangzihan/part_A_cropped/' + apex.strip()
        onset_depth = re.sub('color','depth',re.sub('jpg', 'png', onset))
        apex_depth = re.sub('color','depth',re.sub('jpg', 'png', apex))
        img_onset = cv2.imread(onset)
        img_onset = cv2.resize(img_onset,(224,224)).astype(np.float32)
        img_onset = img_onset/255.0
        img_onset = np.swapaxes(img_onset, 0, 2)
        img_onset = np.swapaxes(img_onset, 1, 2)
        #img_apex = cv2.imread(apex)

        img_onset_depth = cv2.imread(onset_depth, 2)
        img_apex_depth = cv2.imread(apex_depth, 2)
        img_onset_depth = cv2.resize(img_onset_depth,(224,224)).astype(np.float32)
        img_apex_depth = cv2.resize(img_apex_depth,(224,224)).astype(np.float32)
        img_onset_depth = np.expand_dims(img_onset_depth, axis=0)
        img_apex_depth = np.expand_dims(img_apex_depth, axis=0)
        depth_diff = (img_apex_depth-img_onset_depth)/1500.0
        # if self.transform is not None:
        #     img_onset = self.transform(img_onset)
        #     img_apex = self.transform(img_apex)
        #     all = torch.cat((img_onset, img_apex), dim=0)
        #     if self.transform_norm is not None and self.train:
        #         all = self.transform_norm(all)
        #     img_onset = all[0:3,:,:]
        #     img_apex = all[3:6,:,:]
        emo = np.argmax(emo)

        return img_onset, depth_diff, emo

class CASME3_RGBD(data.Dataset):
    def __init__(self,root_path,train,num_loso,transform=None,basic_aug=False,transform_norm=None):
        self.root_path = root_path
        self.train = train
        self.num_loso = num_loso
        self.transform = transform
        self.transform_norm = transform_norm
        self.basic_aug = basic_aug
        self.subject_out_idx = num_loso
        self.img_path = open(os.path.join('/NAS_REMOTE/wangzihan', 'list6/CASME_onset_img_path.txt')).readlines()
        self.idx = []

        for i in range(len(self.img_path)):
            num = self.img_path[i].split('/')[0]
            num = int(re.sub('spNO.', '', num))
            if num == self.subject_out_idx:
                self.idx.append(i)
        if len(self.idx):
            start = self.idx[0]
            end = self.idx[-1]
        else:
            end,start = -1,0
        self.onset = open(os.path.join(self.root_path, 'list6', 'CASME_onset_img_path.txt')).readlines()
        self.apex = open(os.path.join(self.root_path, 'list6', 'CASME_apex_img_path.txt')).readlines()
        self.emotion = np.loadtxt(os.path.join(self.root_path, 'list6', 'CASME_emo_label.txt'))

        if self.train:
            train_onset = self.onset[:start] + self.onset[end+1:]
            train_apex = self.apex[:start] + self.apex[end+1:]
            train_emo = np.delete(self.emotion, self.idx, 0)
            self.data_list = make_dataset(train_onset, train_apex, train_emo)
        else:
            test_onset = self.onset[start:end+1]
            test_apex = self.apex[start:end+1]
            test_emo = self.emotion[self.idx, :]
            self.data_list = make_dataset(test_onset, test_apex, test_emo)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        onset, apex, emo = self.data_list[item]
        onset = '/data2/wangzihan/part_A_cropped/' + onset.strip()
        apex = '/data2/wangzihan/part_A_cropped/' + apex.strip()
        img_onset = cv2.imread(onset)
        img_apex = cv2.imread(apex)
        img_onset = cv2.resize(img_onset, (224, 224)).astype(np.float32)
        img_apex = cv2.resize(img_apex, (224, 224)).astype(np.float32)
        img_onset = img_onset/255.0
        img_onset = np.swapaxes(img_onset, 0, 2)
        img_onset = np.swapaxes(img_onset, 1, 2)
        img_apex = img_apex/255.0
        img_apex = np.swapaxes(img_apex, 0, 2)
        img_apex = np.swapaxes(img_apex, 1, 2)

        onset_depth = re.sub('color','depth',re.sub('jpg', 'png', onset))
        apex_depth = re.sub('color','depth',re.sub('jpg', 'png', apex))

        img_onset_depth = cv2.imread(onset_depth, 2)
        img_apex_depth = cv2.imread(apex_depth, 2)
        img_onset_depth = cv2.resize(img_onset_depth,(224,224)).astype(np.float32)
        img_apex_depth = cv2.resize(img_apex_depth,(224,224)).astype(np.float32)
        img_onset_depth = np.expand_dims(img_onset_depth, axis=0)
        img_apex_depth = np.expand_dims(img_apex_depth, axis=0)
        depth_diff = (img_apex_depth-img_onset_depth)/1500.0

        # if self.transform is not None:
        #     img_onset = self.transform(img_onset)
        #     img_apex = self.transform(img_apex)
        #     all = torch.cat((img_onset, img_apex), dim=0)
        #     if self.transform_norm is not None and self.train:
        #         all = self.transform_norm(all)
        #     img_onset = all[0:3,:,:]
        #     img_apex = all[3:6,:,:]
        img_onset = torch.concat((torch.tensor(img_onset), torch.tensor(img_onset_depth)/1500.0), dim=0)
        img_apex = torch.concat((torch.tensor(img_apex), torch.tensor(img_apex_depth) / 1500.0), dim=0)
        emo = np.argmax(emo)

        return img_onset, img_apex, emo

if __name__ == '__main__':
    train_dataset = CASME3_RGBD('/NAS_REMOTE/wangzihan/', train=True, num_loso=1)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32,
                                               num_workers=2,
                                               shuffle=True,
                                               pin_memory=True)
    for onset,of,emo in train_loader:
        print(onset.shape,of.shape,emo.shape)