import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.utils

import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pickle
import torchvision.utils as vutils
import os, sys
import shutil
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
join=os.path.join

parser = argparse.ArgumentParser()

parser.add_argument('--n', type=int, default=100)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--runs', type=int, default=2)
args = parser.parse_args()

np.random.seed(args.seed)

batch_size = 100
weight_decay = 0.0
learning_rate = 5e-3
lr_step = 1000
do_perm = True
max_data_size = args.n
log_dir = args.log_dir

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

batch_size = min(max_data_size, batch_size)

img_size = 32
flattened_size = img_size * img_size

# device = torch.device("cuda:0")


fname = join(log_dir, 'mnist_result_' + str(max_data_size) + '.txt')

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, flattened_size))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
    
    
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, flattened_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(model, optimizer, train_loader, iteration):
    model.train()
    train_loss = 0.0
    epochs = 500
    for ep in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            data = data[0].to(device)
            data = data.view((-1, flattened_size))
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var)

            loss.backward()
            if ep == epochs - 1:
                train_loss += loss.item()
            optimizer.step()
        
    return train_loss / (len(train_loader.dataset))

def in_top_k(scores, k):
    indices = np.argsort(scores)[::-1]
    pos = np.where(indices==0)[0][0]
    return pos <= 5


def evaluate():
    scores = []
    real_data = data1 = datasets.MNIST(args.data_dir, train=False, download=False, transform=
            transforms.Compose([transforms.Resize(28), transforms.ToTensor()])).test_data.type(torch.float32)/ 255.
    Fake_MNIST = pickle.load(open('Fake_MNIST_data_EP100_N10000.pckl', 'rb'))

    for rep in range(100):
        indices = torch.randperm(real_data.size(0))[:max_data_size]
        data1 = real_data[indices]

        data1 = F.interpolate(torch.unsqueeze(data1, axis=1), img_size)
        ind_tr = np.random.choice(4000, max_data_size, replace=False)   
        data2 = (torch.from_numpy(Fake_MNIST[0][ind_tr]) + 1.0)/2.0
        data2 = data2.type(torch.float32)
        # data2 = F.interpolate(data2, img_size)

        data1 = torch.squeeze(data1, axis=1)
        data2 = torch.squeeze(data2, axis=1)

        if rep != 0:
            data_all = torch.cat([data1, data2], axis=0)
            data_all = data_all[np.random.permutation(range(data_all.shape[0]))]
            data1 = data_all[:max_data_size]
            data2 = data_all[max_data_size:]

        train_data1 = torch.utils.data.TensorDataset(data1)
        train_loader1 = torch.utils.data.DataLoader(train_data1, batch_size=batch_size, shuffle=True)

        train_data2 = torch.utils.data.TensorDataset(data2)
        train_loader2 = torch.utils.data.DataLoader(train_data2, batch_size=batch_size, shuffle=True)

        train_datam = torch.utils.data.TensorDataset(torch.cat([data1[:max_data_size // 2], data2[:max_data_size // 2]], axis=0))
        train_loaderm = torch.utils.data.DataLoader(train_datam, batch_size=batch_size, shuffle=True)


        model = VAE(x_dim=flattened_size, h_dim1= 128, h_dim2=64, z_dim=8).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss2 = train(model=model, optimizer=optimizer, train_loader=train_loader2, iteration=rep)
        lossm = train(model=model, optimizer=optimizer, train_loader=train_loaderm, iteration=rep)
        loss1 = train(model=model, optimizer=optimizer, train_loader=train_loader1, iteration=rep)

        vdiv = np.mean(lossm) - min(np.mean(loss1), np.mean(loss2))

        print('Iteration - ', rep, ' vdiv - ', vdiv)
        scores.append(vdiv)
    return in_top_k(scores, k=5)


def get_scores():   
    evaluations = 100
    print('total runs - ', args.runs)
    across_runs_test_power = []
    for run in range(args.runs):
        print('Starting run - ', run)
        test_power_val = []
        for evaluation in range(evaluations):
            test_power_val.append(evaluate()*1.0)
            with open(fname, 'a') as f:
                f.write(str(evaluation) + " " + str(np.mean(test_power_val)) + '\n')
            print('Iteration ', evaluation, '/', evaluations, ' Average Test power - ', np.mean(test_power_val))

        with open(fname, 'a') as f:
            f.write(str(np.mean(test_power_val)) + " " + str(np.std(test_power_val)) + '\n')
        print(np.mean(test_power_val), np.std(test_power_val))
        across_runs_test_power.append(np.mean(test_power_val))
        print(across_runs_test_power)
    print('across ', args.runs, ' runs the mean test power is ', np.mean(across_runs_test_power))

if __name__ == "__main__":

    get_scores()








