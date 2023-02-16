import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = './materials/'
ROOT_CSV = './materials/'

dataset_type="miniimagenet"
#dataset_type="gesture10"
#dataset_type="gesture26"

if dataset_type=="miniimagenet":
    ROOT_PATH = '../all-datasets/datasets/miniimagenet_origin'
    ROOT_CSV = '../all-datasets/datasets/miniimagenet_origin/split'
elif dataset_type=="gesture10":
    ROOT_PATH = '../all-datasets/datasets/miniimagenet_gesture_test10'
    ROOT_CSV='../all-datasets/datasets/miniimagenet_gesture_test10/split'
elif dataset_type=="gesture26":
    ROOT_PATH = '../all-datasets/datasets/miniimagenet_gesture_test26'
    ROOT_CSV='../all-datasets/datasets/miniimagenet_gesture_test26/split'

class MiniImageNet(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(ROOT_CSV, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print("dataset:",setname)
        #print("data_num:",len(self.data))
        #print("data:",self.data)
        print("data_num:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

