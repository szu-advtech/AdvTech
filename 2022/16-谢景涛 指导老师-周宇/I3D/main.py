# This is a sample Python script.
import numpy as np
import torch

from i3dmodel.I3D import I3D
import os
import random
import shutil
import time
import math
import warnings
from torch.nn import DataParallel
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data




# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# print("Initiating Model...")
# model1=I3D(3,40)
# input_date=np.random.random(size=(1,3,64,224,224))
# x=torch.tensor(input_date)
# print(x.shape)
# print(type(x))
# pre=model1(x.float())



#
import torch
import numpy as np
from torch.optim import lr_scheduler

import  os

from dataloader.dataloader import videoDataset
from i3dmodel.I3D import I3D

print("Using cuda...")
os.environ["CUDA_VISIBLE_DEVICES"] = '3,6,7'

torch.backends.cudnn.benchmark = True

# Parameters
print("Creating params....")
params = {'batch_size':4,
          'shuffle': True,
          'num_workers': 2}
params2 = {'batch_size': 2,
           'shuffle': False,
           'num_workers': 2}
max_epochs = 110000

train_data='/data/xiejingtao/NTU/train_data.npy'
train_label='/data/xiejingtao/NTU/train_label.npy'
test_data='/data/xiejingtao/NTU/test_data.npy'
test_label='/data/xiejingtao/NTU/test_label.npy'
print("Creating Data Generators...")
training_set=videoDataset(train_data,train_label,'train')
training_generator=torch.utils.data.DataLoader(training_set,**params)
validation_set=videoDataset(test_data,test_label,'test')
validation_generator=torch.utils.data.DataLoader(validation_set, **params2)
print("Initiating Model...")
model1=I3D(3,60)
model_path = 'pretrainstep.pth'
model_data = torch.load(model_path)
model1.load_state_dict(model_data,strict=False)
model1=model1.cuda()
model1=DataParallel(model1)
criterion=torch.nn.CrossEntropyLoss()
lr=0.01
wt_decay=5e-4
optimizer = torch.optim.SGD(model1.parameters(), lr=lr,momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=10)


# epoch_loss_train=[]
# epoch_loss_val=[]
# epoch_acc_train=[]
# epoch_acc_val=[]
best_accuracy = 0.
print("Begin Training....")
for epoch in range(max_epochs):
    
    model1.train()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    ##

    ##
    number=0
    for inputs,targets in training_generator:
        inputs=inputs.cuda()
        targets=targets.cuda()
        optimizer.zero_grad()
        predictions=model1(inputs.float())
        batch_loss=criterion(predictions,targets)
        batch_loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss += batch_loss.sum().item()
            accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
        cnt += len(targets)
        number+=len(targets)
        if number>1000:
            print(f"Epoch: {epoch},train:{cnt}/22000")
            number=0
    loss /= cnt
    accuracy *= 100. / cnt
    print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
    #epoch_loss_train.append(loss)
    #epoch_acc_train.append(accuracy)
        # scheduler.step()

        # accuracy,loss = validation(model,validation_generator)
    scheduler.step(loss)
        # Test
    
    model1.eval()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    model1 = model1.cuda()
    with torch.no_grad():
        for inputs, targets in validation_generator:
            b = inputs.shape[0]
            inputs = inputs.cuda()  # print("Validation input: ",inputs)
            targets = targets.cuda()

            predictions = model1(inputs.float())
            batch_loss=criterion(predictions,targets)
            with torch.no_grad():
                    loss += batch_loss.sum().item()
                    accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
            
        loss /= cnt
        accuracy *= 100. / cnt

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            # torch.save(model1.state_dict(),  '_best_ckpt.pt')
            # print('Check point  _best_ckpt.pt Saved!')

    print(f"Epoch: {epoch},Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")

    #epoch_loss_val.append(loss)
    #epoch_acc_val.append(accuracy)
print(f"best_aaccuracy{best_accuracy}")



# torch.save(model.state_dict(), "./data/model_parameter.pkl")
# new_model = Model()                                                    # 调用模型Model
# new_model.load_state_dict(torch.load("./data/model_parameter.pkl"))    # 加载模型参数
# new_model(input)
