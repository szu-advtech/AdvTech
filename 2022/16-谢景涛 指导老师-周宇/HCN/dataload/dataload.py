import numpy as np
import torch
import torch.nn.functional as F


def poseRandomCrop(body,min_ratio=0.5,max_ratio=1.0,min_len=64):
    num_frames=body.shape[1]
    min_frames=int(num_frames* min_ratio)
    max_frames = int(num_frames * max_ratio)
    clip_len = np.random.randint(min_frames, max_frames + 1)
    clip_len = min(max(clip_len, min_len), num_frames)
    start = np.random.randint(0, num_frames - clip_len + 1)
    inds = np.arange(start, start + clip_len)
    body=body[:,inds,:,:]
    return body
def poseCenterCrop(body,clip_ratio=0.9):
    num_frames = body.shape[1]
    clip_len=int(num_frames*clip_ratio)
    start = (num_frames - clip_len) // 2
    inds = np.arange(start, start + clip_len)
    body=body[:,inds,:,:]
    return body
def PosrResize(body,clip_len=32):
    m, t, v, c = body.shape
    body = body.transpose((0, 3, 1, 2))  # M T V C -> M C T V
    body = F.interpolate(
        torch.from_numpy(body),
        size=(clip_len, v),
        mode='bilinear',
        align_corners=False)
    body = body.permute((0, 2, 3, 1))
    return body

def precessdata(body,flag):
    if flag=='train':
        body=poseRandomCrop(body)
    else:
        body=poseCenterCrop(body)
    
    body=PosrResize(body)
    
    return body

class posesDataset(torch.utils.data.Dataset):
    def __init__(self,train_data,train_label,flag):

        self.train_data = np.load(train_data,allow_pickle=True).item()
        self.train_label = np.load(train_label,allow_pickle=True).item()
        self.len = len(self.train_label)
        self.flag=flag

    def __getitem__(self, index):
        with open(self.train_data[index], 'r') as fs:
            frames_number = int(fs.readline())
            body1 = []
            body2 = []
            for i in range(frames_number):
                frames_body = int(fs.readline())
                zeo=[]
                numberp=frames_body
                for k in range(frames_body):
                    fs.readline()
                    joint_number = int(fs.readline())
                    jo = []
                    zeo=[]
                    for j in range(joint_number):
                        temp = fs.readline()
                        t = temp.split(' ')
                        x, y, z = t[0:3]
                        x = float(x)
                        y = float(y)
                        z = float(z)
                        jo.append([x, y, z])
                        zeo.append([0,0,0])
                    if k==0 and numberp==1:
                        body1.append(jo)
                        body2.append(zeo)
                    elif k==0 and numberp>=2:
                        body1.append(jo)
                    elif k==1 and numberp>=2:
                        body2.append(jo)
                    else:
                        continue


        
            frames=[body1,body2]
            frames= np.array(frames, dtype=float)
            #print(frames.shape)
            #print(self.train_data[index])
            frames = precessdata(frames,self.flag)
            X =frames
            Y=self.train_label[index]

        return X,Y

    def __len__(self):
        return self.len
