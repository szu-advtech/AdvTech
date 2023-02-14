from osgeo import gdal
import os
import numpy as np
import cv2
import random

from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, UpSampling2D
from keras.optimizers import Adam

#from keras.engine.input_layer import Input
from keras.layers.merging.concatenate import concatenate

import os
import datetime
import xlwt
import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint


def readTif(fileName):
    """
    读取图像的像素矩阵,这里为了能支持多波段,我们利用GDAL读取
    argus:
        fileName 图像文件名
    """
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount        # 波段数
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    print(GdalImg_data)
    return GdalImg_data


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


def color_dict(labelFolder, classNum):
    """
    color_dict函数除了返回colorDict_GRAY,还会返回colorDict_RGB,用于预测时RGB渲染显示;
    #  获取颜色字典
    #  labelFolder 标签文件夹,之所以遍历文件夹是因为一张标签可能不包含所有类别颜色
    #  classNum 类别总数(含背景)
    """
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if(len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if(len(colorDict) == classNum):
            break
    #  存储颜色的RGB字典，用于预测时的渲染结果
    colorDict_RGB = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位R,中3位G,后3位B
        color_RGB = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
        colorDict_RGB.append(color_RGB)
    #  转为numpy格式
    colorDict_RGB = np.array(colorDict_RGB)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_RGB.reshape(
        (colorDict_RGB.shape[0], 1, colorDict_RGB.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_RGB, colorDict_GRAY

def trainGenerator(batch_size, train_image_path, train_label_path, classNum, colorDict_GRAY, resize_shape = None):
    """ 训练数据生成器
    利用keras.Model.fit_generator()函数进行训练,
    所以我们需要一个训练数据生成器,keras自带的生成器不支持多波段,所以我们自己编写实现
    #  batch_size 批大小
    #  train_image_path 训练图像路径
    #  train_label_path 训练标签路径
    #  classNum 类别总数(含背景)
    #  colorDict_GRAY 颜色字典
    #  resize_shape resize大小
    """
    imageList = os.listdir(train_image_path)
    labelList = os.listdir(train_label_path)
    img = readTif(train_image_path + "\\" + imageList[0])
    #  GDAL读数据是(BandNum,Width,Height)要转换为->(Width,Height,BandNum)
    img = img.swapaxes(1, 0)
    img = img.swapaxes(1, 2)
    #  无限生成数据
    while(True):
        img_generator = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        label_generator = np.zeros((batch_size, img.shape[0], img.shape[1]), np.uint8)
        if(resize_shape != None):
            img_generator = np.zeros((batch_size, resize_shape[0], resize_shape[1], resize_shape[2]), np.uint8)
            label_generator = np.zeros((batch_size, resize_shape[0], resize_shape[1]), np.uint8)
        #  所有图像中随机生成一个batch的起点
        rand = random.randint(0, len(imageList) - batch_size)
        for j in range(batch_size):
            img = readTif(train_image_path + "\\" + imageList[rand + j])
            img = img.swapaxes(1, 0)
            img = img.swapaxes(1, 2)
            #  改变图像尺寸至特定尺寸(
            #  因为resize用的不多，我就用了OpenCV实现的，这个不支持多波段，需要的话可以用np进行resize
            if(resize_shape != None):
                img = cv2.resize(img, (resize_shape[0], resize_shape[1]))
            img_generator[j] = img
            
            label = readTif(train_label_path + "\\" + labelList[rand + j]).astype(np.uint8)
            #  若为彩色，转为灰度
            if(len(label.shape) == 3):
                label = label.swapaxes(1, 0)
                label = label.swapaxes(1, 2)
                label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            if(resize_shape != None):
                label = cv2.resize(label, (resize_shape[0], resize_shape[1]))
            label_generator[j] = label
        img_generator, label_generator = dataPreprocess(img_generator, label_generator, classNum, colorDict_GRAY)
        yield (img_generator,label_generator)

def unet(pretrained_weights=None, input_size=(256, 256, 4), classNum=2, learning_rate=1e-5):
    inputs = Input(input_size)
    #  2D卷积层
    conv1 = BatchNormalization()(Conv2D(64, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(inputs))
    conv1 = BatchNormalization()(Conv2D(64, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(conv1))
    #  对于空间数据的最大池化
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = BatchNormalization()(Conv2D(128, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(pool1))
    conv2 = BatchNormalization()(Conv2D(128, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = BatchNormalization()(Conv2D(256, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(pool2))
    conv3 = BatchNormalization()(Conv2D(256, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = BatchNormalization()(Conv2D(512, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(pool3))
    conv4 = BatchNormalization()(Conv2D(512, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(conv4))
    #  Dropout正规化，防止过拟合
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = BatchNormalization()(Conv2D(1024, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(pool4))
    conv5 = BatchNormalization()(Conv2D(1024, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(conv5))
    drop5 = Dropout(0.5)(conv5)
    #  上采样之后再进行卷积，相当于转置卷积操作
    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))

    try:
        merge6 = concatenate([drop4, up6], axis=3)
    except:
        merge6 = concatenate([drop4, up6], axis=3)
    conv6 = BatchNormalization()(Conv2D(512, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(merge6))
    conv6 = BatchNormalization()(Conv2D(512, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(conv6))

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    try:
        merge7 = concatenate([conv3, up7], axis=3)
    except:
        merge7 = concatenate([conv3, up7], axis=3)
    conv7 = BatchNormalization()(Conv2D(256, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(merge7))
    conv7 = BatchNormalization()(Conv2D(256, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(conv7))

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    try:
        merge8 = concatenate([conv2, up8], axis=3)
    except:
        merge8 = concatenate([conv2, up8], axis=3)
    conv8 = BatchNormalization()(Conv2D(128, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(merge8))
    conv8 = BatchNormalization()(Conv2D(128, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(conv8))

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    try:
        merge9 = concatenate([conv1, up9], axis=3)
    except:
        merge9 = concatenate([conv1, up9], axis=3)
    conv9 = BatchNormalization()(Conv2D(64, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(merge9))
    conv9 = BatchNormalization()(Conv2D(64, 3, activation='relu', padding='same',
                                        kernel_initializer='he_normal')(conv9))
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(classNum, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    #  用于配置训练模型（优化器、目标函数、模型评估标准）
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    #  如果有预训练的权重
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


if __name__ == "__main__":
    #  训练数据图像路径
    train_image_path = "C:\\Users\\James\\Desktop\\project\\data\\train\\img"
    #  训练数据标签路径
    train_label_path = "C:\\Users\\James\\Desktop\\project\\data\\train\\label"
    #  验证数据图像路径
    validation_image_path = "C:\\Users\\James\\Desktop\\project\\data\\validation\\img"
    #  验证数据标签路径
    validation_label_path = "C:\\Users\\James\\Desktop\\project\\data\\validation\\label"

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
    img = readTif(train_image_path + "\\" + imageList[0])
    





'''
#  得到一个生成器，以batch_size的速率生成训练数据
train_Generator = trainGenerator(batch_size,
                                 train_image_path,
                                 train_label_path,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)

#  得到一个生成器，以batch_size的速率生成验证数据
validation_data = trainGenerator(batch_size,
                                 validation_image_path,
                                 validation_label_path,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)
#  定义模型
model = unet(pretrained_weights=premodel_path,
             input_size=input_size,
             classNum=classNum,
             learning_rate=learning_rate)
#  打印模型结构
model.summary()
#  回调函数
model_checkpoint = ModelCheckpoint(model_path,
                                   monitor='loss',
                                   verbose=1,  # 日志显示模式:0->安静模式,1->进度条,2->每轮一行
                                   save_best_only=True)

#  获取当前时间
start_time = datetime.datetime.now()

#  模型训练
history = model.fit_generator(train_Generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              callbacks=[model_checkpoint],
                              validation_data=validation_data,
                              validation_steps=validation_steps)

#  训练总时间
end_time = datetime.datetime.now()
log_time = "训练总时间: " + str((end_time - start_time).seconds / 60) + "m"
print(log_time)
with open('TrainTime.txt', 'w') as f:
    f.write(log_time)
'''