import cv2
import numpy as np
import torch
def imgResizeandcrrop(frames):
    N,H,W,C=frames.shape
    rd=np.random.randint(0,2)
    new_h = 224
    new_w = 224
    new_frames = np.zeros((N, 224, 224, 3), dtype=np.float32)
    y = np.random.randint(0, 256 - new_h)
    x = np.random.randint(0, 256 - new_w)
    for i in range(N):
        t=cv2.resize(frames[i],(256,256))  
        new_frames[i] =t[y:y+new_h,x:x+new_w]
        new_frames[i] = cv2.flip(new_frames[i], rd)
    return new_frames
def precesstraindata(path):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(path)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    for fr_idx in range(num_frames):
        ret, frame = cap.read()
        if frame is None:
            continue
        frames.append(frame)
    frames = np.array(frames,dtype=np.float32)
    num_frames=frames.shape[0]
    if num_frames>64:
        start = np.random.randint(0, num_frames - 64 + 1)
        inds = np.arange(start, start + 64)
        frames=frames[inds]
        frames=imgResizeandcrrop(frames)
        frames = torch.from_numpy(frames)
        frames = frames.permute(3, 0, 1, 2)
        return  frames
    else:
        frames = imgResizeandcrrop(frames)
        #补充
        data = np.zeros((64, 224, 224, 3))
        a=(64-num_frames)//2
        data[a:a+num_frames]=frames
        data = data.astype(np.float32)
        data = torch.from_numpy(data)
        data = data.permute(3, 0, 1, 2)
        return data
def precesstestdata(path):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    new_frame = np.zeros((300, 224, 224, 3), dtype=np.float32)
    new_h = 224
    new_w = 224
    y = (256 - 224) // 2
    x = y
    i=0
    for fr_idx in range(num_frames):
        ret, frame = cap.read()
        if frame is None:
            continue
        frame = cv2.resize(frame, (256, 256))
        new_frame[i] = frame[y:y+new_h,x:x+new_w]
        i+=1
    frames = new_frame
    frames = np.array(frames, dtype=np.float32)
    frames = torch.from_numpy(frames)
    frames = frames.permute(3, 0, 1, 2)
    return frames



class videoDataset(torch.utils.data.Dataset):
    def __init__(self,data,label,flag):
        self.data=np.load(data,allow_pickle=True).item()
        self.label=np.load(label,allow_pickle=True).item()
        self.len=len(self.data)
        self.flag=flag

    def __getitem__(self, item):
        path=self.data[item][1]
        if self.flag=='train':
            X=precesstraindata(path)
        elif self.flag=='test':
            X=precesstestdata(path)
        Y=self.label[item]
        return X,Y

    def __len__(self):
        return self.len
