import numpy as np
import torch
import torch.nn.functional as F
import cv2

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

def imgResizeandcrrop(frames):
    N, H, W, C = frames.shape
    rd = np.random.randint(0, 2)
    new_frames = np.zeros((N, 224, 224, 3), dtype=np.float32)
    for i in range(N):
        new_h = 224
        new_w = 224
        y = np.random.randint(0, 256 - new_h)
        x = np.random.randint(0, 256 - new_w)
        t = cv2.resize(frames[i], (256, 256))

        new_frames[i] = t[y:y + new_h, x:x + new_w]
        new_frames[i] = cv2.flip(new_frames[i], rd)
    return new_frames
def precesstraindata(path):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for fr_idx in range(num_frames):
        ret, frame = cap.read()
        frames.append(frame)
    frames = np.array(frames)
    if num_frames>64:
        start = np.random.randint(0, num_frames - 64 + 1)
        inds = np.arange(start, start + 64)
        frames=frames[inds]
        frames = np.array(frames, dtype=np.float32)
        frames=imgResizeandcrrop(frames)
        frames = frames.astype(np.float32)
        frames = torch.from_numpy(frames)
        frames = frames.permute(3, 0, 1, 2)
        return  frames
    else:
        frames = np.array(frames, dtype=float)
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
    for fr_idx in range(num_frames):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (256, 256))
        new_frame[fr_idx] = frame[y:y + new_h, x:x + new_w]
    frames = new_frame
    frames = np.array(frames, dtype=np.float32)
    frames = torch.from_numpy(frames)
    frames = frames.permute(3, 0, 1, 2)
    return frames


class posesDataset(torch.utils.data.Dataset):
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
            # print(frames.shape)
            # print(self.train_data[index])
            frames = precessdata(frames, self.flag)
            X1 = frames
            path = self.train_data[index][1]
            if self.flag == 'train':
                X2 = precesstraindata(path)
            else:
                X2 = precesstestdata(path)
            Y = self.train_label[index]
           
            return (X1,X2),Y

    def __len__(self):
        return self.len




