"""
@Name: Train.py
@Auth: SniperIN_IKBear
@Date: 2022/11/30-21:03
@Desc: 
@Ver : 0.0.0
"""
#%%
import argparse
import copy
import os

# import spectral
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from torch import nn, optim
import scipy.io as sio


from HybridSN.Model.HybridSN_Attention_class import HybridSN_Attention
from HybridSN.Utills.MultGPUs import dataparallel
from HybridSN.Utills.MyDataset import load_set, TrainDS, TestDS, predicted_set

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


## Model Config
# parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
# parser.add_argument('--data_path', default='/home2/szx/Dataset/TreeDetection/DHIF/Train/', type=str,
#                     help='Path of the training data')
# parser.add_argument("--sizeI", default=96, type=int, help='The image size of the training patches')
# parser.add_argument("--batch_size", default=16, type=int, help='Batch size')
# parser.add_argument("--trainset_num", default=20000, type=int, help='The number of training samples of each epoch')
# parser.add_argument("--sf", default=8, type=int, help='Scaling factor')
# parser.add_argument("--seed", default=1, type=int, help='Random seed')
# parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
# parser.add_argument("--testset_num", default=1, type=int, help='total number of testset')
# opt = parser.parse_args()
#%%
## Model Config
parser = argparse.ArgumentParser(description="PyTorch Code HybridSN")
parser.add_argument('--class_num', default='8', type=str,help='num for types')
Xtrain, Xtest, ytrain, ytest = load_set()
trainset = TrainDS(Xtrain,ytrain)
testset = TestDS(Xtest,ytest)
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=False, num_workers=2)

def mytest_acc(net):
  count = 0
  # 模型测试
  for inputs, _ in test_loader:
      inputs = inputs.cuda()
      outputs = net(inputs)
      outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
      if count == 0:
          y_pred_test =  outputs
          count = 1
      else:
          y_pred_test = np.concatenate( (y_pred_test, outputs) )

  # 生成分类报告
  classification = classification_report(ytest, y_pred_test, digits=4)
  index_acc = classification.find('weighted avg')
  accuracy = classification[index_acc+17:index_acc+23]
  return float(accuracy)

def train(net):


    current_loss_his = []
    current_Acc_his = []

    best_net_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 开始训练
    total_loss = 0
    for epoch in range(150):
      net.train()  # 将模型设置为训练模式
      for i, (inputs, labels) in enumerate(train_loader):
          inputs = inputs.cuda()
          labels = labels.cuda()
          # 优化器梯度归零
          optimizer.zero_grad()
          # 正向传播 +　反向传播 + 优化
          outputs = net(inputs)
          ####################################################################
          # predict_image = spectral.imshow(classes=outputs.detach().cpu().numpy().astype(int), figsize=(5, 5))
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          total_loss += loss.item()

      net.eval()   # 将模型设置为验证模式
      current_acc = mytest_acc(net)
      current_Acc_his.append(current_acc)

      if current_acc > best_acc:
        best_acc = current_acc
        best_net_wts = copy.deepcopy(net.state_dict())
        torch.save(best_net_wts,'/home2/szx/Dataset/TreeDetection/CheckPointHybridSNSZU/attention_best_net_wts_%d.th'%(epoch))
      print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]  [current acc: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item(), current_acc))
      current_loss_his.append(loss.item())

    print('Finished Training')
    print("Best Acc:%.4f" %(best_acc))

    # load best model weights
    net.load_state_dict(best_net_wts)

    return net,current_loss_his,current_Acc_his

# # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 网络放到GPU上
net = dataparallel(HybridSN_Attention(),3)
# 训练
net,current_loss_his,current_Acc_his = train(net)


def show_plot(data,plot_name,x_name,y_name):
  plt.title(plot_name)
  plt.xlabel(x_name)
  plt.ylabel(y_name)
  plt.plot(data)
  plt.savefig(r'./{}.jpg'.format(plot_name))
  plt.close()


show_plot(current_loss_his,'Hybrid train loss plot','Epoch','Loss')
show_plot(current_Acc_his,'Hybrid Accuracy plot','Epoch','Accuracy')

#测试
net.eval()   # 将模型设置为验证模式
# 测试最好的模型的结果
count = 0
# 模型测试
list_outs=np.random.random((128,8)) ####
for inputs, _ in test_loader:
    inputs = inputs.cuda()
    outputs = net(inputs)

    list_outs = np.concatenate([list_outs, outputs.detach().cpu().numpy()], axis=0)
    outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
    if count == 0:
        y_pred_test =  outputs
        count = 1
    else:
        y_pred_test = np.concatenate( (y_pred_test, outputs) )

# 生成分类报告
classification = classification_report(ytest, y_pred_test, digits=4)
print(classification)
list_outs = np.array(list_outs).astype(int)
#整张图测试：
# inputs_pre = predicted_set()
# inputs_pre = inputs_pre.cuda()
# imgout = net(inputs_pre)
# predict_image = spectral.imshow(classes=list_outs,figsize=(5,5))
# print(12331212)