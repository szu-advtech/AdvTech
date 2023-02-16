"""
@Name: predicted-txt.py
@Auth: SniperIN_IKBear
@Date: 2023/1/31-13:00
@Desc: 
@Ver : 0.0.0
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from sklearn.metrics import classification_report
import spectral
from torch import nn
import os


from HybridSN.Model.HybridSN_Attention_class import HybridSN_Attention
from HybridSN.Utills.Get3DPatch import applyPCA, padWithZeros
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
pca_components = 30
patch_size = 25

def pred_net(net, device, X,y):
    # 加载训练集
    X = applyPCA(X, numComponents=pca_components)
    X = padWithZeros(X, patch_size // 2)
    height = y.shape[0]
    width = y.shape[1]
    acc = 0
    # net.load_state_dict(torch.load('best_model_net_BN_Attention.pth', map_location=device))  # 加载模型参数
    #net.load_state_dict(torch.load(r'./Checkpoint/attention_best_net_wts_113.th', map_location=device))  # 加载模型参数

      # 实例化自己的模型；
    checkpoint = torch.load(r'/home2/szx/Dataset/TreeDetection/CheckPointHybridSNSZU/attention_best_net_wts_66.th', map_location=device)
    # checkpoint = torch.load(r'/home2/szx/Project/TreeDetection/HybridSN/Checkpoint/attention_best_net_wts_80.th',
    #                         map_location=device)
    # 加载模型文件，pt, pth 文件都可以；
    if torch.cuda.device_count() > 1:
        # 如果有多个GPU，将模型并行化，用DataParallel来操作。这个过程会将key值加一个"module. ***"。
        net = nn.DataParallel(net)
    net.load_state_dict(checkpoint)  # 接着就可以将模型参数load进模型。

    net.eval()
    # 逐像素预测类别
    outputs = np.zeros((height, width))
    count = 0
    y_lable = []
    y_pre = []
    for i in range(height):
        for j in range(width):
            # if int(y[i, j]) == 0:
            #     continue
            # else:
            #     image_patch = X[i:i + patch_size, j:j + patch_size, :]
            #     image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
            #                                       1)
            #     X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
            #     prediction = net(X_test_image)
            #     prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            #
            #     y_pre.append(prediction[0]+1)
            #     y_lable.append(y[i,j])
            #
            #
            #
            #     outputs[i][j] = prediction + 1

            image_patch = X[i:i + patch_size, j:j + patch_size, :]
            image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                              1)
            X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
            prediction = net(X_test_image)
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction + 1
            if int(y[i, j]) != 0:
                y_pre.append(prediction[0] + 1)
                y_lable.append(y[i,j])

        if i % 20 == 0:
            print('... ... row ', i, ' handling ... ...')

    def Normalize(data):
        h, w= data.shape
        data = data.reshape((h * w))
        data -= np.min(data, axis=0)
        data /= np.max(data, axis=0)
        data = data.reshape((h, w))
        return data

    sio.savemat('/home2/szx/Dataset/TreeDetection/CheckPointHybridSNSZU/outputs.mat', {'img': outputs})
    outputs = Normalize(outputs)
    sio.savemat('/home2/szx/Dataset/TreeDetection/CheckPointHybridSNSZU/outputs_Normalize.mat',{'img':outputs})
    plt.imshow(outputs)
    plt.show()


    classification = classification_report(np.array(y_lable),np.array(y_pre), digits=4)
    print(classification)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    X = sio.loadmat('/home2/szx/Dataset/TreeDetection/HybridSN/hyper_lidar_szu.mat')['data']#[:,240:,:]
    y = sio.loadmat('/home2/szx/Dataset/TreeDetection/HybridSN/true_label_szu.mat')['AllTrueLabel']#[:,240:]


    # 数据归一化
    def Normalize(data):
        h, w, c = data.shape
        data = data.reshape((h * w, c))
        data -= np.min(data, axis=0)
        data /= np.max(data, axis=0)
        data = data.reshape((h, w, c))
        return data


    # 数据归一化并缩放到[-1, 1]
    X = 2 * Normalize(X) - 1

    net = HybridSN_Attention()
    # net  =HybridSN_BN_Attention()
    # 将网络拷贝到deivce中
    net.to(device=device)
    pred_net(net,device,X,y)




