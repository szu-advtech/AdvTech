import torch
import numpy as np
from torch.optim import lr_scheduler
from torch.nn import DataParallel
from dataload.dataload import posesDataset

from model.HCN import hcn
import  os
print("Using cuda...")
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
print("Creating params....")
params = {'batch_size':1024,
          'shuffle': True,
          'num_workers': 16}
max_epochs = 1000
train_data='/data/xiejingtao/NTU/S1_17/train_data.npy'
train_label='/data/xiejingtao/NTU/S1_17/train_label.npy'
test_data='/data/xiejingtao/NTU/S1_17/test_data.npy'
test_label='/data/xiejingtao/NTU/S1_17/test_label.npy'
print("Creating Data Generators...")
training_set=posesDataset(train_data,train_label,'train')
training_generator=torch.utils.data.DataLoader(training_set,**params)
validation_set=posesDataset(test_data,test_label,'test')
validation_generator=torch.utils.data.DataLoader(validation_set, **params)
print("Initiating Model...")
model1=hcn(3,25,60)
mode_data=torch.load('best_ckpt2.pt')
model1.load_state_dict(mode_data,strict=False)
model1=model1.cuda()
model1=DataParallel(model1)
criterion=torch.nn.CrossEntropyLoss()
lr=0.0001
wt_decay=5e-4
def add_weight_decay(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if name=='fc7.weight' or name=='fc7.bias' : decay.append(param)
        else: no_decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 0.001}]
parameters=add_weight_decay(model1);
optimizer = torch.optim.Adam(parameters, lr=lr,betas=(0.9, 0.999),eps=1e-08)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)


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
    scheduler.step()
    model1.eval()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    model1 = model1.to(device)
    with torch.no_grad():
        for inputs, targets in validation_generator:
            b = inputs.shape[0]
            inputs = inputs.cuda()  # print("Validation input: ",inputs)
            targets = targets.cuda()

            predictions = model1(inputs.float())

            with torch.no_grad():
                    loss += batch_loss.sum().item()
                    accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        loss /= cnt
        accuracy *= 100. / cnt

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model1.state_dict(),  'best_ckpt.pt')
            # print('Check point  _best_ckpt.pt Saved!')

    print(f"Epoch: {epoch},Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")

    #epoch_loss_val.append(loss)
    #epoch_acc_val.append(accuracy)
print(f"best_aaccuracy{best_accuracy}")
