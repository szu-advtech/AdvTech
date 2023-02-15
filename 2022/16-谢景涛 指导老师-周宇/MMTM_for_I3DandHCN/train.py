
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# print("Initiating Model...")
# model1=mmtm(3,25,3,60)
# video_date=np.random.random(size=(8,3,64,224,224))
# skeleten=np.random.random(size=(8,2,32,25,3))
# video_date=torch.tensor(video_date).float()
# skeleten=torch.tensor(skeleten).float()
# x=[video_date,skeleten]
# pre=model1(x)
# print(pre)
import torch
from model.MMTM import mmtm
from dataset.dataset import posesDataset
import  os
print("Using cuda...")
os.environ["CUDA_VISIBLE_DEVICES"] = '0,2'

torch.backends.cudnn.benchmark = True

# Parameters
print("Creating params....")
params = {'batch_size':8,
          'shuffle': True,
          'num_workers': 4}
params2 = {'batch_size': 2,
           'shuffle': False,
           'num_workers': 2}
max_epochs = 110000
#dataset
train_data='/data/xiejingtao/NTU/train_data.npy'
train_label='/data/xiejingtao/NTU/train_label.npy'
test_data='/data/xiejingtao/NTU/test_data.npy'
test_label='/data/xiejingtao/NTU/test_label.npy'
print("Creating Data Generators...")
training_set=posesDataset(train_data,train_label,'train')
training_generator=torch.utils.data.DataLoader(training_set,**params)
validation_set=posesDataset(test_data,test_label,'test')
validation_generator=torch.utils.data.DataLoader(validation_set, **params2)
print("Initiating Model...")
model=mmtm(3,25,3,60)
model_path = '/data/xiejingtao/I3D/pretrainstep.pth'
model_data = torch.load(model_path)
model.load_state_dict(model_data,strict=False)

model=model.cuda()
criterion=torch.nn.CrossEntropyLoss()
#训练初始化
lr=0.01
wt_decay=5e-4
optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=6)


#训练开始
best_accuracy = 0.
print("Begin Training....")
for epoch in range(max_epochs):
    model.train()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    for inputs,targets in training_generator:
        inputs =[ input.cuda() for input in inputs]
        targets=targets.cuda()
        optimizer.zero_grad()
        predictions=model([input.float() for input in inputs])
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
            video=inputs[:][1]
            skeleton=inputs[:][0]
            depth=depth.cuda()
            skeleton=skeleton.cuda()
            targets = targets.cuda()

             predictions = model([skeleton.float(),video.float()])

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
    torch.save(checkpoint, '/data/xiejingtao/MMTM/model.pth')
    print(f"Epoch: {epoch},Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")
print(f"best_aaccuracy{best_accuracy}")






