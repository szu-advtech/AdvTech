#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' export CUDA_LAUNCH_BLOCKING=1')


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from matplotlib import pyplot as plt
import torch
import logging
import os
import h5py
import numpy as np
import pandas as pd
import random
from torch import nn
import torchvision
import torch.nn.functional as F
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

logger = logging.getLogger(__name__)
if logger.handlers:
    logger.handlers = []
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())        


# In[4]:


device = torch.device('cuda')
torch.cuda.get_device_properties(0)


# In[5]:


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


# In[6]:


def median_smoothing(signal, width=3):
    s = signal[:]
    half_width = int(width/2)
    for i in range(half_width, len(signal)):
        s[i-half_width] = torch.median(signal[i-half_width:i+half_width])
    return s


# In[7]:


class MolbitDataset(Dataset):
    def __init__(self, data_file):
        with h5py.File(data_file, "r") as f:
            self.data = torch.FloatTensor(f.get("data")[()])
            self.labels = torch.IntTensor(f.get("labels")[()])
        
        self.n_records = len(self.labels)
        self.max_len = self.data.shape[2]
        self.n_labels = len(np.unique(self.labels))

        # Shuffle data
        self.shuffle_index = np.random.choice(range(self.n_records), replace=False, size=self.n_records)
        self.data = self.data[self.shuffle_index]       
        self.labels = self.labels[self.shuffle_index]        
        
    def _get_onehot(self, label):
        ix = self.labels.index(label)
        onehot = torch.zeros(self.n_labels)
        onehot[ix] = 1
        return onehot
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.data[idx, :, :], self.labels[idx]


# In[8]:


molbit_data_file = "/path/to/data/v4_train/20190822_train.hdf5"


# In[9]:


md = MolbitDataset(molbit_data_file)


# In[13]:


md.n_records


# In[11]:


sns.distplot(md.labels, kde=False)


# In[10]:


molbit = 82
max_len = 3000

ixs_of_molbit = np.where(md.labels == molbit)[0]
for ix in ixs_of_molbit[:10]:
    data, label = md[ix]
    fig, ax = plt.subplots(figsize=(20, 3))
    
    data = median_smoothing(data, width=5)
    ax.plot(range(max_len), np.array(data.squeeze()[:max_len]), c="k")
    ax.set_title(str(molbit))
    ax.set_ylim([-2, 3])
    ax.axhline(0, c="k")
    plt.show()


# In[11]:


sum(md.labels == 82)


# In[10]:


class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        O_1 = 64
        O_2 = 128
        O_3 = 256
        O_4 = 512
        O_5 = 1024

        K_1 = 15
        K_2 = 8
        K_3 = 6
        K_4 = 4
        K_5 = 2

        KP_1 = 6
        KP_2 = 3
        KP_3 = 2
        KP_4 = 2
        KP_5 = 1

        FN_1 = 1000
        FN_2 = 500

        self.conv1 = nn.Sequential(nn.Conv1d(1, O_1, K_1, stride=1), nn.ReLU(),
                                   nn.AvgPool1d(KP_1))
        self.conv1_bn = nn.BatchNorm1d(O_1)

        self.conv2 = nn.Sequential(nn.Conv1d(O_1, O_2, K_2), nn.ReLU(),
                                   nn.AvgPool1d(KP_2))
        self.conv2_bn = nn.BatchNorm1d(O_2)

        self.conv3 = nn.Sequential(nn.Conv1d(O_2, O_3, K_3), nn.ReLU(),
                                   nn.AvgPool1d(KP_3))
        self.conv3_bn = nn.BatchNorm1d(O_3)

        self.conv4 = nn.Sequential(nn.Conv1d(O_3, O_4, K_4), nn.ReLU(),
                                   nn.AvgPool1d(KP_4))
        self.conv4_bn = nn.BatchNorm1d(O_4)

        self.conv5 = nn.Sequential(nn.Conv1d(O_4, O_5, K_5), nn.ReLU(),
                                   nn.AvgPool1d(KP_5))
        self.conv5_bn = nn.BatchNorm1d(O_5)

        self.gru1 = nn.GRU(input_size=92160, hidden_size=10, num_layers=1)

#         self.fc1 = nn.Linear(conv_linear_out, FN_1, nn.Dropout(0.4))
        self.fc1 = nn.Linear(37888, FN_1, nn.Dropout(0.5)) # 37888 20480 28672 9216
        self.fc1_bn = nn.BatchNorm1d(FN_1)

        self.fc2 = nn.Linear(FN_1, FN_2, nn.Dropout(0.5))
        self.fc2_bn = nn.BatchNorm1d(FN_2)

        self.fc3 = nn.Linear(FN_2, 96)

    def forward(self, x):
        x = x.float()
#         print("a", np.shape(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_bn(x))
#         print("b", np.shape(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_bn(x))
#         print("c", np.shape(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3_bn(x))
#         print("d", np.shape(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.relu(self.conv4_bn(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.relu(self.conv5_bn(x))
#         print("e", np.shape(x))
    #	  x = x.view(1, 1, -1)
    #	  x = self.gru1(x)
        x = x.view(len(x), -1)
#         print("f", np.shape(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1_bn(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_bn(x))
#         print("g", np.shape(x))

        x = self.fc3(x)
#         print("h", np.shape(x))
        return x


net = CNN()


# In[24]:


seed_everything()

n_labels = md.n_labels
model = CNN()
model.cuda()


loader_params = {"batch_size": 20,
          "shuffle": False,
          "num_workers": 20}
max_epochs = 200

# Datasets
n_samples = len(md)
all_ix = range(n_samples)
n_train = int(0.84*n_samples)
n_val = n_samples - n_train
train_ix = np.random.choice(all_ix, size=n_train)
val_ix = np.array(list(set(all_ix[:]) - set(train_ix)))
train_sampler = SubsetRandomSampler(train_ix)
val_sampler = SubsetRandomSampler(val_ix)
training_generator = DataLoader(md, sampler=train_sampler, **loader_params)
validation_generator = DataLoader(md, sampler=val_sampler, **loader_params)


# In[39]:


optim_params = {"lr": 0.001,
                "momentum": 0.5}

optimizer = torch.optim.Adam(model.parameters())#, **optim_params)
loss_fn = torch.nn.CrossEntropyLoss()

train_losses = []
val_losses = []
train_accs = []
val_accs = []

val_delta = 0.01
patience = 30

max_val_acc = 0
last_val_change = 0

logger.log(10, "Starting")

for epoch in range(max_epochs):
    logger.log(10, "Epoch: %d" % epoch)
    if last_val_change >= patience:
        print("Patience ran out.")
        break

    train_total_samples = 0
    train_correct = 0
    val_total_samples = 0
    val_correct = 0
    
    logger.log(10, "Train")
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        logger.log(1, "Transferring data to GPU")
        local_batch = local_batch.to(device)
        local_labels = local_labels.to(device)
        
#         print(local_batch.shape)
#         print(local_labels.shape)
        
        logger.log(1, "Run model")
        optimizer.zero_grad()
        y_pred_train = model(local_batch).to(dtype=torch.float64)
#         print(y_pred_train.shape)
        local_labels = local_labels.to(dtype=torch.long)
        _, ypred_max_ix = torch.max(y_pred_train, dim=1)
        
        logger.log(1, "Calculate loss")
        train_loss = loss_fn(y_pred_train, local_labels)
        logger.log(1, "Backprop")
        train_loss.backward()
        optimizer.step()

        train_total_samples += local_labels.size(0)
        train_correct += (local_labels == ypred_max_ix).sum().item()

    logger.log(10, "Val")
    # Validation
    with torch.no_grad():
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch = local_batch.to(device)
            local_labels = local_labels.to(device)
            y_pred_val = model(local_batch).to(dtype=torch.float64)
            local_labels = local_labels.to(dtype=torch.long)
            
            _, ypred_max_ix = torch.max(y_pred_val, dim=1)

            val_loss = loss_fn(y_pred_val, local_labels)

            val_total_samples += local_labels.size(0)
            val_correct += (local_labels == ypred_max_ix).sum().item()

    train_acc = 100. * train_correct / train_total_samples
    val_acc = 100. * val_correct / val_total_samples
    print("%d: Train loss: %0.7f\tVal loss: %0.7f\tTrain acc: %2.5f%%\tVal acc: %2.5f%%"
          % (epoch, train_loss.item(), val_loss.item(), train_acc, val_acc))
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if val_acc - max_val_acc <= val_delta:
        last_val_change += 1
    else:
        last_val_change = 0
    max_val_acc = max(val_acc, max_val_acc)


# In[40]:


plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()


# In[41]:


plt.plot(train_accs, label="train")
plt.plot(val_accs, label="val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%%)")
plt.legend()


# In[42]:


np.max(train_accs), np.max(val_accs)


# ## Save the model

# In[27]:


get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(`notebook_name = \'${window.document.getElementById("notebook_name").innerHTML}\'`);')


# In[63]:


todays_date = datetime.today().strftime('%Y%m%d')
torch.save(model.state_dict(), f"saved_models/{notebook_name}.{todays_date}.pt")


# In[16]:


model_file = "/ssd1/home/kdorosch/code/punchcard-tagger/v4/molbit_classification/saved_models/molbit_classification_v4_0_1.20190827.pt"
model = CNN()
model.load_state_dict(torch.load(model_file))
model.eval()
model.cuda()


# ## Load the model

# In[17]:


model_file = "/ssd1/home/kdorosch/code/punchcard-tagger/v4/molbit_classification/saved_models/molbit_classification_v4_0_1.20190827.pt"


# In[18]:


model = CNN()
model.load_state_dict(torch.load(model_file))
model.cuda()
model.eval()


# # Evaluate error

# In[19]:


from sklearn.metrics import confusion_matrix


# In[25]:


val_loader_params = {"batch_size": 500,
          "shuffle": False,
          "num_workers": 30}
validation_generator = DataLoader(md, sampler=val_sampler, **val_loader_params)


# In[26]:


y_pred_all = []
y_true_all = []
for i, (local_batch, local_labels) in enumerate(validation_generator):
#     print(i)
#     print(local_labels.shape)
    # Transfer to GPU
    local_batch = local_batch.to(device)
    local_labels = local_labels.to(device)
    y_pred_val = model(local_batch).to(dtype=torch.float64)
    local_labels = local_labels.to(dtype=torch.long)
    
    t, u = torch.max(y_pred_val, dim=1)
    y_pred_molbits = [int(x) for x in u]
    y_true_molbits = [int(x) for x in local_labels.cpu()]

    y_pred_all.extend(y_pred_molbits)
    y_true_all.extend(y_true_molbits)


# In[27]:


np.sum([x == y for x, y in zip(y_true_all, y_pred_all)]), len(y_true_all)


# In[28]:


1. * np.sum([x == y for x, y in zip(y_true_all, y_pred_all)]) / len(y_true_all)


# In[29]:


l, c = np.unique(y_true_all, return_counts=True)


# In[30]:


cm = confusion_matrix(y_true_all, y_pred_all)


# In[31]:


fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(cm / c[:, np.newaxis], vmin=0, vmax=0.01)#, vmax=1)
c
plt.savefig(f"/ssd1/home/kdorosch/code/punchcard-tagger/v4/plots/by_model/{notebook_name}_cm_validation.svg")


# In[ ]:


from scipy.stats import pearsonr

norm_cm = cm / c[:, np.newaxis]
plt.scatter(c, np.diag(norm_cm))
plt.xlabel("#/ validation examples")
plt.ylabel("Accuracy")
pearsonr(c, np.diag(norm_cm))


# 
# # Make predictions on full set

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


loader_params = {"batch_size": 300,
          "shuffle": True,
          "num_workers": 30}
data_generator = DataLoader(md, **loader_params)


# In[ ]:


len(y_pred_all)


# In[ ]:


np.unique(md.labels, return_counts=True)


# In[ ]:


np.unique(y_true_all)


# In[ ]:


y_pred_all = []
y_true_all = []
for i, (local_batch, local_labels) in enumerate(data_generator):
#     print(i)
#     print(local_labels.shape)
    # Transfer to GPU
    local_batch = local_batch.to(device)
#     local_labels = local_labels.to(device)
    y_pred_val = model(local_batch).to(dtype=torch.float64)
#     local_labels = local_labels.to(dtype=torch.long)
    
    t, u = torch.max(y_pred_val, dim=1)
    y_pred_molbits = [int(x) for x in u]
    y_true_molbits = [int(x) for x in local_labels.cpu()]

    y_pred_all.extend(y_pred_molbits)
    y_true_all.extend(y_true_molbits)


# In[ ]:


np.sum([x == y for x, y in zip(y_true_all, y_pred_all)]), len(y_true_all)


# In[ ]:


1. * np.sum([x == y for x, y in zip(y_true_all, y_pred_all)]) / len(y_true_all)


# In[ ]:


l, c = np.unique(y_true_all, return_counts=True)


# In[ ]:


cm = confusion_matrix(y_true_all, y_pred_all)


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(cm / c[:, np.newaxis], vmin=0, vmax=0.01)#, vmax=1)
ax.set_title("Confusion matrix for model v4_0_1\npredictions across all training data")
ax.set_xlabel("True (by sequence)")
ax.set_ylabel("Predicted (using model on squiggles)")
plt.savefig(f"/ssd1/home/kdorosch/code/punchcard-tagger/v4/plots/by_model/{notebook_name}_cm_alltrain.svg")


# In[ ]:


from scipy.stats import pearsonr

norm_cm = cm / c[:, np.newaxis]
plt.scatter(c, np.diag(norm_cm))
plt.xlabel("#/ training examples")
plt.ylabel("Accuracy")
pearsonr(c, np.diag(norm_cm))


# 
