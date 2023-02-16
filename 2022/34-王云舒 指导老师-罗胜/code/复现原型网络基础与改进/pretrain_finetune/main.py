import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
from NetsAndFunc_lib import get_acc
from sklearn.metrics import confusion_matrix
from operate_Func import plot_Matrix
device = torch.device("cuda")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
classes=["0","1","2","3","4","5","6","7","8","9"]

# 训练epoch次
def train(epoch):
    model.train()  # 训练模式
    for batch_idx, (img, label) in enumerate(trainloader):
        #print(img)
        #print(label)
        img = np.array(img).astype(float)
        img = torch.from_numpy(img).float()
        label = np.array(label).astype(int)
        label = torch.from_numpy(label)
        image = Variable(img.cuda())  # 放到gpu上
        label = Variable(label.cuda())  # 放到gpu上
        # image = Variable(img)
        # label = Variable(label)
        optimizer.zero_grad()  # 意思是把梯度置零，也就是把loss关于weight的导数变成0.
        out = model(image)  # 投喂图片
        loss = criterion(out, label)  # 利用交叉熵损失函数算出out和label的差别
        loss.backward()  # 反向传播
        optimizer.step()
        train_acc = get_acc(out, label)  # 获得准确率
        print("Epoch:%d [%d|%d] loss:%f acc:%f" % (epoch + 1, batch_idx, len(trainloader), loss.mean(), train_acc))
    scheduler.step()  # 按照Pytorch的定义是用来更新优化器的学习率的

# 验证准确率
def test(epoch):
    count = 0
    y_true = []
    y_pred = []
    print("Validation Epoch: %d" % (epoch + 1))
    model.eval()  # 进入推理模式
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(testloader):
            img = np.array(img).astype(float)
            img = torch.from_numpy(img).float()
            label = np.array(label).astype(int)
            label = torch.from_numpy(label)
            image = Variable(img.cuda())
            label = Variable(label.cuda())
            # image = Variable(img)
            # label = Variable(label)
            out = model(image)
            _, predicted = torch.max(out.data, 1)
            total += image.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()
            for i in range(image.size(0)):
                y_true.append(label[i].cpu().numpy())
                y_pred.append(predicted[i].cpu().numpy())
                count = count + 1
            if epoch==1 and batch_idx==1:
                plt.figure()
                plt.tight_layout()
                this_image = img[0]
                this_image = this_image.swapaxes(0, 1)
                this_image = this_image.swapaxes(1, 2)
                plt.imshow(this_image)
                if label[i]==predicted[i]:
                    plt.title("Truth:"+str(label[i]) + " Predict:"+str(predicted[i]), color='blue')
                else:
                    plt.title("Truth:"+str(label[i]) + " Predict:"+str(predicted[i]), color='red')
                #plt.savefig('handwrite_result_save/Test_show.jpg')
                plt.show()
    print("accuracy: %f " % ((1.0 * correct.numpy()) / total))
    print("total_num:" + str(count))
    cm = confusion_matrix(y_true, y_pred)
    #plot_Matrix(epoch, cm, classes, title=None, cmap=plt.get_cmap('Purples'))



import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
########################################Only ResNet12; Get pretrained model########################################
#ROOT_PATH = './materials/'
#ROOT_PATH = '../../all-datasets/datasets/miniimagenet_gesture_test10'
#ROOT_CSV='../../all-datasets/datasets/miniimagenet_gesture_test10/split'
ROOT_PATH = '../../all-datasets/datasets/miniimagenet_origin'
ROOT_CSV='../../all-datasets/datasets/miniimagenet_origin/split'
class MiniImageNet(Dataset):

    def __init__(self, setname, train, test):
        csv_path = osp.join(ROOT_CSV, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.train = train
        self.test = test
        self.data = []
        self.label = []
        lb = -1
        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.label.append(lb)

        imgs_num=len(self.data)
        random.seed(30)  # SEED= this must be opened
        random.shuffle(self.data)
        random.seed(30)  # SEED= this must be opened
        random.shuffle(self.label)
        if self.train:
            self.data = self.data[:int(0.8 * imgs_num)]
            self.label = self.label[:int(0.8 * imgs_num)]
        elif self.test:
            self.data = self.data[int(0.8 * imgs_num):]
            self.label = self.label[int(0.8 * imgs_num):]
        print(self.label)
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

trainset = MiniImageNet('train', train=True, test=False)
testset = MiniImageNet('train', train=False, test=True)
print("——————————————————————————这是分割线——————————————————————————")
print(trainset.__len__())
print(testset.__len__())
print("——————————————————————————这是分割线——————————————————————————")
# 继承了dataset                            dataset     一捆有多大      数据顺序      多线程输入，=0表示单线程
batch_size_define=32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_define, shuffle=False, num_workers=0)#这里shuffle也可以改成True来乱序
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_define, shuffle=False, num_workers=0)

device = torch.device("cuda")
import resnet_new
from resnet12new import resnet12_wide
from resnet12 import ResNet12
import torchvision.models as models
from NetsAndFunc_lib import Net_resnet_pretrained
#model = resnet_new.resnet18()
#model = models.resnet18(num_classes=64)
model = ResNet12(output_size=64,avg_pool=True, drop_rate=0.0).to('cuda')#新baseline下的resnet12
#model = resnet12_wide().to('cuda')#新baseline下的resnet12
print(model)

# USEPreModel=True
# pretrain_kernel_size =5
# pretrain_trainways = 64
# if USEPreModel == True:
#     model = models.resnet18(pretrained=True)
#     if pretrain_kernel_size == 5:
#         model.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1, bias=False)  # 5x5
#     model.fc = nn.Linear(512, pretrain_trainways)
#     print(model)

model = model.to(device)  # 转到gpu/cpu上运行
#---model.parameters()为当前的网络模型的参数空间，lr=为学习率↓
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)  # 设置训练细节
scheduler = StepLR(optimizer, step_size=3)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
for epoch in range(10):
    train(epoch)
    test(epoch)
torch.save(model.state_dict(), 'miniimagenet.pkl')  # 保存模型
model.to("cpu")  # 转换回cpu