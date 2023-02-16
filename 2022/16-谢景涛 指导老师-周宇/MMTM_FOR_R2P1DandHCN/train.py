import torch
from model.MMTM import mmtm
from dataloader.dataset import MyDataset
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
print("Using cuda...")
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'

torch.backends.cudnn.benchmark = True

# Parameters
print("Creating params....")
params = {'batch_size':64,
          'shuffle': True,
          'num_workers': 4}
params2 = {'batch_size': 8,
           'shuffle': False,
           'num_workers': 2}
max_epochs = 110000
#dataset
train_data='/data/xiejingtao/NTU/depth_skeleten_cs/train_data.npy'
train_label='/data/xiejingtao/NTU/depth_skeleten_cs/train_label.npy'
test_data='/data/xiejingtao/NTU/depth_skeleten_cs/test_data.npy'
test_label='/data/xiejingtao/NTU/depth_skeleten_cs/test_label.npy'
print("Creating Data Generators...")
training_set=MyDataset(train_data,train_label,'train')
training_generator=torch.utils.data.DataLoader(training_set,**params)
validation_set=MyDataset(test_data,test_label,'test')
validation_generator=torch.utils.data.DataLoader(validation_set, **params2)
print("Initiating Model...")
model=mmtm(3,25,60)
modelr3d_path = '/data/xiejingtao/MMTM2/r3dpretrain.pt'
modelhcn_path= '/data/xiejingtao/MMTM2/hcnpretrain.pt'
modelr3d_data = torch.load(modelr3d_path)
modelhcn_data = torch.load(modelhcn_path)
model.load_state_dict(modelr3d_data,strict=False)
model.load_state_dict(modelhcn_data,strict=False)

model=model.cuda()
model=DataParallel(model)
criterion=torch.nn.CrossEntropyLoss()
#训练初始化
lr=0.01
wt_decay=5e-4
optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=10)


#训练开始
best_accuracy = 0.
print("Begin Training....")
for epoch in range(max_epochs):
    model.train()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    for inputs,targets in training_generator:
        depth=inputs[:][1]
        skeleton=inputs[:][0]
        depth=depth.cuda()
        skeleton=skeleton.cuda()
        targets=targets.cuda()
        optimizer.zero_grad()
        predictions=model([skeleton.float(),depth.float()])
        batch_loss=criterion(predictions,targets)
        batch_loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss += batch_loss.sum().item()
            accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
        cnt += len(targets)
    loss /= cnt
    accuracy *= 100. / cnt
    print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
    scheduler.step(loss)
        # Test
    model.eval()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    model = model.cuda()
    with torch.no_grad():
        for inputs, targets in validation_generator:
            depth=inputs[:][1]
            skeleton=inputs[:][0]
            bn, length_test, chanl,t, w, h = depth.shape
            depth = depth.reshape(bn * length_test, chanl,t, w, h)
            skeleton = torch.repeat_interleave(skeleton, length_test, dim=0)
            # print("Validation input: ",inputs)
            depth=depth.cuda()
            skeleton=skeleton.cuda()
            targets = targets.cuda()
            predictions = model([skeleton.float(),depth.float()])
            predictions = predictions.reshape(bn, length_test, -1)
            predictions = predictions.mean(1)
            batch_loss = criterion(predictions, targets)
            with torch.no_grad():
                    loss += batch_loss.sum().item()
                    accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        loss /= cnt
        accuracy *= 100. / cnt
        if best_accuracy < accuracy:
            best_accuracy = accuracy
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), \
              'epoch': epoch, 'lr': scheduler.state_dict(),}
    torch.save(checkpoint, '/data/xiejingtao/MMTM2/model2.pth')
    print(f"Epoch: {epoch},Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")
print(f"best_aaccuracy{best_accuracy}")