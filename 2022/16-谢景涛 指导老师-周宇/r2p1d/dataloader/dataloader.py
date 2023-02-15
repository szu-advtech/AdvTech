import cv2
import numpy as np
import torch
import os


def crop(path,rd,x,y):
    data = cv2.imread(path, -1)
    data = data[~np.all(data == 0, axis=1)]
    data = np.delete(data, np.where(~data.any(axis=0))[0], axis=1)
    data = cv2.resize(data, (256, 256))
    data =data[y:y+224,x:x+224]
    data = cv2.flip(data, rd)
    data=np.expand_dims(data,axis=2)
    return data

def precesstrain(path):
    list=os.listdir(path)
    length=len(list)
    frames=np.zeros((8, 224, 224, 1), dtype=np.float32)
    rd=np.random.randint(0,2)
    y = np.random.randint(0, 256 - 224)
    x = np.random.randint(0, 256 - 224)
    '''
    if length<=30:
        for i in range(30):
            if i<length:
                frames[i] = crop(path+'/' + list[i])
            else:
                frames[i] = frames[i-1]
    elif length<=60:
        start = np.random.randint(0, length - 30 + 1)
        j=0
        for i in range(start,start+30):
            frames[j] = crop(path+'/' + list[i])
            j+=1

    else:
        start = np.random.randint(0, length - 60 + 1)
        j=0
        for i in range(start,start+60,2):
            frames[j] = crop(path+'/' + list[i])
            j+=1
    '''
    number=length//8
    for i in range(8):
        n = np.random.randint(number*i, min(number*(i+1),length))
        frames[i]=crop(path+'/' + list[n],rd,x,y)
    frames = torch.from_numpy(frames)
    frames = frames.permute(3, 0,1, 2)
    return frames


def precesstest(path):
    list=os.listdir(path)
    length=len(list)
    frames=np.zeros((10,8, 224, 224, 1), dtype=np.float32)
    number=length//8
    y = (256 - 224) // 2
    x = y
    for n in range(10):
      for i in range(8):
          t = np.random.randint(number*i, min(number*(i+1),length))
          frames[n,i]=crop(path+'/' + list[t],0,x,y)
    frames = torch.from_numpy(frames)
    frames = frames.permute(0, 4,  1,2, 3)
    return frames


class depthDataset(torch.utils.data.Dataset):
    def __init__(self,data,label,flag):
        self.data = np.load(data, allow_pickle=True).item()
        self.label = np.load(label, allow_pickle=True).item()
        self.len = len(self.data)
        self.flag = flag
        
    def __getitem__(self, item):
        path=self.data[item]
        if self.flag=='train':
            X=precesstrain(path)
        else:
            X=precesstest(path)
        Y=self.label[item]
        return X,Y
    
    def __len__(self):
        return self.len



