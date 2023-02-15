import torch
import numpy as np
from torch.optim import lr_scheduler
from torch.nn import DataParallel
import  os

from dataloader.dataloader import depthDataset
from model.model import r2plus1d18
print("Using cuda...")
os.environ["CUDA_VISIBLE_DEVICES"] = '5,6,7'
torch.backends.cudnn.benchmark = True

# Parameters
print("Creating params....")
params = {'batch_size':125,
          'shuffle': True,
          'num_workers': 2}
params2 = {'batch_size': 16,
           'shuffle': False,
           'num_workers': 2}
max_epochs = 1000
train_data='/data/xiejingtao/NTU/depth/depthcs/train_data.npy'
train_label='/data/xiejingtao/NTU/depth/depthcs/train_label.npy'
test_data='/data/xiejingtao/NTU/depth/depthcs/test_data.npy'
test_label='/data/xiejingtao/NTU/depth/depthcs/test_label.npy'
print("Creating Data Generators...")
training_set=depthDataset(train_data,train_label,'train')
training_generator=torch.utils.data.DataLoader(training_set,**params)
validation_set=depthDataset(test_data,test_label,'test')
validation_generator=torch.utils.data.DataLoader(validation_set, **params2)
print("Initiating Model...")
model1=r2plus1d18(num_classes=60)

model_datapath = 'model1_13.pth'

model_data = torch.load(model_datapath)

model1.load_state_dict(model_data,strict=False)

model1=model1.cuda()
model1=DataParallel(model1)
criterion=torch.nn.CrossEntropyLoss()


lr=0.001
optimizer = torch.optim.SGD(model1.parameters(), lr=lr,momentum=0.9)
# CLR






best_accuracy = 0.
print("Begin Training....")
for epoch in range(max_epochs):
    
    model1.train()
    loss = 0.
    accuracy = 0.
    cnt = 0.
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
    loss /= cnt
    accuracy *= 100. / cnt
    print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
    
    
# Test
    model1.eval()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    model1 = model1.cuda()
    with torch.no_grad():
        for inputs, targets in validation_generator:
            bn,length_test,chanl,w,h,t = inputs.shape
            inputs=inputs.reshape(bn*length_test,chanl,w,h,t)
            inputs = inputs.cuda()  # print("Validation input: ",inputs)
            
            targets = targets.cuda()
            predictions = model1(inputs.float())
            predictions=predictions.reshape(bn,length_test,-1)
            
            predictions=predictions.mean(1)

            batch_loss=criterion(predictions,targets)
            with torch.no_grad():
                    loss += batch_loss.sum().item()
                    accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        loss /= cnt
        accuracy *= 100. / cnt

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model1.module.state_dict(),  'model1_14.pth')
            # print('Check point  _best_ckpt.pt Saved!')

    print(f"Epoch: {epoch},Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")

print(f"best_aaccuracy{best_accuracy}")




