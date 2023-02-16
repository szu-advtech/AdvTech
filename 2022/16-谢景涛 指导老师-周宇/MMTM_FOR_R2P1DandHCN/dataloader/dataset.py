import cv2
import numpy as np
import torch
import os
import torch.nn.functional as F

def crop(path, rd, x, y):
    data = cv2.imread(path, -1)
    data = data[~np.all(data == 0, axis=1)]
    data = np.delete(data, np.where(~data.any(axis=0))[0], axis=1)
    data = cv2.resize(data, (256, 256))
    data = data[y:y + 224, x:x + 224]
    data = cv2.flip(data, rd)
    data = np.expand_dims(data, axis=2)
    return data


def precesstrain(path):
    list = os.listdir(path)
    length = len(list)
    frames = np.zeros((8, 224, 224, 1), dtype=np.float32)
    rd = np.random.randint(0, 2)
    y = np.random.randint(0, 256 - 224)
    x = np.random.randint(0, 256 - 224)
    number = length // 8
    for i in range(8):
        n = np.random.randint(number * i, min(number * (i + 1), length))
        frames[i] = crop(path + '/' + list[n], rd, x, y)
    frames = torch.from_numpy(frames)
    frames = frames.permute(3, 0,1, 2)
    return frames


def precesstest(path):
    list = os.listdir(path)
    length = len(list)
    frames = np.zeros((10, 8, 224, 224, 1), dtype=np.float32)
    number = length // 8
    y = (256 - 224) // 2
    x = y
    for n in range(10):
        for i in range(8):
            t = np.random.randint(number * i, min(number * (i + 1), length))
            frames[n, i] = crop(path + '/' + list[t], 0, x, y)
    frames = torch.from_numpy(frames)
    frames = frames.permute(0, 4,  1,2, 3)
    return frames

def poseRandomCrop(body, min_ratio=0.5, max_ratio=1.0, min_len=64):
    num_frames = body.shape[1]
    min_frames = int(num_frames * min_ratio)
    max_frames = int(num_frames * max_ratio)
    clip_len = np.random.randint(min_frames, max_frames + 1)
    clip_len = min(max(clip_len, min_len), num_frames)
    start = np.random.randint(0, num_frames - clip_len + 1)
    inds = np.arange(start, start + clip_len)
    body = body[:, inds, :, :]
    return body
def poseCenterCrop(body, clip_ratio=0.9):
    num_frames = body.shape[1]
    clip_len = int(num_frames * clip_ratio)
    start = (num_frames - clip_len) // 2
    inds = np.arange(start, start + clip_len)
    body = body[:, inds, :, :]
    return body

def PosrResize(body, clip_len=32):
    m, t, v, c = body.shape
    body = body.transpose((0, 3, 1, 2))  # M T V C -> M C T V
    body = F.interpolate(
        torch.from_numpy(body),
        size=(clip_len, v),
        mode='bilinear',
        align_corners=False)
    body = body.permute((0, 2, 3, 1))
    return body




def precessdata(body, flag):
    if flag == 'train':
        body = poseRandomCrop(body)
    elif flag=='test':
        body = poseCenterCrop(body)

    body = PosrResize(body)

    return body

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train_data, train_label, flag):
        self.train_data = np.load(train_data, allow_pickle=True).item()
        self.train_label = np.load(train_label, allow_pickle=True).item()
        self.len = len(self.train_label)
        self.flag = flag

    def __getitem__(self, index):
        with open(self.train_data[index][0], 'r') as fs:
            frames_number = int(fs.readline())
            body1 = []
            body2 = []
            for i in range(frames_number):
                frames_body = int(fs.readline())
                zeo = []
                numberp = frames_body
                for k in range(frames_body):
                    fs.readline()
                    joint_number = int(fs.readline())
                    jo = []
                    zeo = []
                    for j in range(joint_number):
                        temp = fs.readline()
                        t = temp.split(' ')
                        x, y, z = t[0:3]
                        x = float(x)
                        y = float(y)
                        z = float(z)
                        jo.append([x, y, z])
                        zeo.append([0, 0, 0])
                    if k == 0 and numberp == 1:
                        body1.append(jo)
                        body2.append(zeo)
                    elif k == 0 and numberp >= 2:
                        body1.append(jo)
                    elif k == 1 and numberp >= 2:
                        body2.append(jo)
                    else:
                        continue

            frames = [body1, body2]
            frames = np.array(frames, dtype=np.float32)

            frames = precessdata(frames, self.flag)
            X1 = frames
            path = self.train_data[index][1]
            if self.flag == 'train':
                X2 = precesstrain(path)
            else:
                X2 = precesstest(path)
            Y = self.train_label[index]

            return (X1, X2), Y

    def __len__(self):
        return self.len



