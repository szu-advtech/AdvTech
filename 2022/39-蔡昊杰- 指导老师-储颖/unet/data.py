import os
import torch
from osgeo import gdal
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from utils import *


def readTif(fileName, mode=''):
    """
    读取图像的像素矩阵,这里为了能支持多波段,我们利用GDAL读取
    argus:
        fileName 图像文件名
    """
    dataset = gdal.Open(fileName)   # 只能打开地址形式的数据,不能打开dataloader中的数据
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount        # 波段数
    geotrans = dataset.GetGeoTransform()      #  获取仿射矩阵信息
    proj = dataset.GetProjection()      #  获取投影信息
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    #print(len(GdalImg_data[]))
    if(mode == 'Initialization'):
        return width, height, bands, geotrans, proj
    else:
        return GdalImg_data


def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(
        im_height), int(im_bands), datatype)
    if(dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset



def dataPreprocess(img, label, classNum, colorDict_GRAY):
    """ 数据预处理：图像归一化+标签onehot编码
    读取了图像之后就要做预处理：
    1. 对图像进行归一化,这里我们采用最大值归一化,即除以最大值255(对于8bit数据来说):
    2. 对标签进行onehot编码,即将平面的label的每类都单独变成由0和1组成的一层。
    argus:
        img 图像数据
        label 标签数据
        classNum 类别总数(含背景)
        colorDict_GRAY 颜色字典
    """
    img = img / 255.0  # 归一化
    for i in range(colorDict_GRAY.shape[0]):
        label[label == colorDict_GRAY[i][0]] = i
    #  将数据厚度扩展到classNum(包括背景)层
    new_label = np.zeros(label.shape + (classNum,))
    #  将平面的label的每类，都单独变成一层
    for i in range(classNum):
        new_label[label == i, i] = 1
    label = new_label
    return (img, label)


def showDataset(train_loader, img_num):
    for i, data in enumerate(train_loader):
        images, labels = data
        print(i, data[0].shape, data[1].shape)

        # 打印数据集中的图片
        img = torchvision.utils.make_grid(
            images, padding=10).numpy()  # tensor->numpy
        plt.imshow(np.transpose(img, (1, 2, 0)))  # (c,h,w)->(h,w,c)
        plt.show()


#自定义数据集的类
class PVDataset(Dataset):

    def __init__(self, image_path, label_path):
        super().__init__()
        self.img_path = image_path
        self.label_path = label_path
        self.imageList = os.listdir(image_path)
        self.labelList = os.listdir(label_path)

        self.width, self.height, self.bands, self.proj, self.geotrans = readTif(
            image_path + "/" + self.imageList[0], 'Initialization')

        self.len = len(self.imageList)
        print('read dataset successful', self.len)

    def __getitem__(self, index):

        img = readTif(self.img_path + '/' + self.imageList[index])
        label = readTif(self.label_path + '/' + self.labelList[index])
        #print(img)
        #print(self.labelList[index])

        #image = cv2.resize(image, dsize=(416, 416))
        #label = cv2.resize(label, dsize=(416, 416))

        # numpy->tensor
        img = torch.tensor(img)
        label = torch.tensor(label)
        #torch.set_printoptions(profile="full")
        #np.set_printoptions(threshold=np.inf)
        #print(img)
        #print(label)

        img = img / 255.0  # 归一化
        label = label / 255.0
        img.requires_grad = True
        #with open('C:\\Users\\James\\Desktop\\label.txt', 'w') as f:
            #torch.set_printoptions(profile="full")
            #np.set_printoptions(threshold=np.inf)
            #for i in range(len(label.numpy())):
                #f.write(str(label.numpy()[i])+'\n')
        #print(img)
        #print(label)

        #image=image.permute(2,0,1)        #调整图像维度,方便载入model
        return img, label

    def __len__(self):

        return self.len


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class UpSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.layer = nn.Conv2d(channel, channel/2, 1, 1)
    def forward(self, x, downsample_feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, downsample_feature_map), dim=1)

def cat_(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    tmp = F.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))  
    x = torch.cat([tmp, x2], dim=1)
    return x

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)   # 转置卷积
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.Th = nn.Sigmoid()           # 避免输出图像出现负值

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        #up_6 = F.interpolate(c5, scale_factor=2, mode='nearest') # 最近邻插值
        up_6 = self.up6(c5)
        #print(up_6.shape)
        merge6 = cat_(up_6, c4)
        #merge6 = torch.cat([up_6, c4], dim=1)  # (N,C,H,W),dim=1代表Channel
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        #print(c10.shape)
        #二分类在预测时，net()输出先做sigmoid()概率化处理，然后大于0.5为1，小于0.5为0。
        return c10

'''
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):  # 0是表示从0开始
        image, label = data
        device = torch.device(
            "cuda:0"if torch.cuda.is_available() else "cpu")  # 检测是否有GPU加速
        image, label = image.to(device), label.to(device)  # 数据放进GPU里
        optimizer.zero_grad()  # 优化器参数清零

        #forword+backward+update
        image = image.type(torch.FloatTensor)  # 转化数据类型,不转则会报错
        image = image.to(device)
        outputs = model(image)
        loss = criterion(outputs, label.long())  # 进行loss计算

        lll = label.long().cpu().numpy()  # 把label从GPU放进CPU

        loss.backward(retain_graph=True)  # 反向传播(求导)
        optimizer.step()  # 优化器更新model权重

        running_loss += loss.item()  # 收集loss的值

        if batch_idx % 100 == 99:
            print('[epoch: %d,idex: %2d] loss:%.3f' %
                  (epoch+1, batch_idx+1, running_loss/322))  # 训练集的数量,可根据数据集调整
            runing_loss = 0.0  # 收集的loss值清零

        torch.save(model.state_dict(),
                   f='D:/untitled/.idea/SS_torch/weights/SS_weight_3.pth')  # 保存权重
'''

'''
if __name__ == "__main__":
    #  批大小
    batch_size = 2
    #  类的数目(包括背景)
    classNum = 2
    #  模型输入图像大小
    input_size = (200, 200, 3)
    #  训练模型的迭代总轮数
    epochs = 100
    #  初始学习率
    learning_rate = 1e-6
    #  预训练模型地址
    premodel_path = None
    #  训练模型保存地址
    model_path = "C:\\Users\\James\\Desktop\\project\\model\\unet_model.hdf5"

    #  训练数据数目
    train_num = len(os.listdir(train_image_path))
    #  验证数据数目
    #validation_num = len(os.listdir(validation_image_path))
    #  训练集每个epoch有多少个batch_size
    steps_per_epoch = train_num / batch_size
    #  验证集每个epoch有多少个batch_size
    #validation_steps = validation_num / batch_size
    #  标签的颜色字典,每个label类别对应一种颜色,用于onehot编码
    #colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)

    imageList = os.listdir(train_image_path)
    labelList = os.listdir(train_label_path)
    pv_trn = PVDataset(train_image_path, train_label_path)

    train_loader = DataLoader(
        dataset=pv_trn, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    net = Unet(3, 2)

    #showDataset(train_loader, batch_size)
    filename = train_label_path + os.sep + labelList[0]
    x = readTif(filename)
    #print(x)
    print(x.shape)        # (3, 200, 200)
    print(net(x).shape)

    net=Unet(3, 2)
    x = torch.randn(1, 3, 256, 256)
    print(net(x).shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)     # 优化器
    #optimizer = optim.SGD(model.parameters(), lr=0.01)

    device = torch.device(
        "cuda:0"if torch.cuda.is_available() else "cpu")  # 检测是否有GPU加速
    model.to(device)  # 网络放入GPU里加速

    model.load_state_dict(torch.load(
        'D:/untitled/.idea/SS_torch/weights/SS_weight_2.pth'))
'''