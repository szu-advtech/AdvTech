import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer

from tool.parameter import args_parser
from tool.test import test_img
from tool.Net import MlpMnist
from tool.server import FedAvg
from tool.dataset import get_dataset1,get_dataset2


class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, beta=0, delta=0, compute='sub',grad=None):
        for group in self.param_groups:
            i=0
            for p  in group['params']:
                if grad is not None:
                    d_p = grad[i]
                    i+=1
                    if delta != 0:
                        if compute == 'add':
                            p.data = p.data + delta*d_p
                        else:
                            p.data = p.data - delta*d_p
                    elif beta != 0:
                        p.data = p.data - beta * d_p
                    else:
                        p.data = p.data - group['lr']*d_p
                else:
                    d_p = p.grad.data
                    if beta != 0:
                        p.data = p.data - beta*d_p
                    else:
                        p.data = p.data - group['lr']*d_p

def comp(grad2,grad3,delta):
    grad={}
    for i in range(len(grad2)):
        grad[i] = (grad2[i]-grad3[i])/(delta*2)
    return grad

def comp1(grad1,dw,lr):
    grad={}
    for i in range(len(grad1)):
        grad[i] = grad1[i]-dw[i]*lr
    return grad

class Datesetsplit(Dataset):
    def __init__(self,dateset,idxs):
        self.dateset = dateset
        self.idx = list(idxs)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        image,label = self.dateset[self.idx[item]]
        return image,label

class LocalUpdate(object):
    def __init__(self,arg,dateset,idx):#
        self.args = arg
        self.ldr_train = DataLoader(Datesetsplit(dateset,idx),batch_size=self.args.local_bs,shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()

    def train(self,g_net):
        net=copy.deepcopy(g_net)
        net.train()
        ep_loss =[]
        opt = MySGD(net.parameters(),lr=self.args.lr)
        batch_loss=[]
        loader_iter = iter(self.ldr_train)

        for t in range(self.args.local_step):
            net.train()
            temp_params = copy.deepcopy(list(net.parameters()))

            #step1
            try:
                batch_x, batch_y = next(loader_iter)
            except Exception:
                loader_iter = iter(self.ldr_train)
                batch_x, batch_y = next(loader_iter)
            batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)
            y_pred = net(batch_x)
            opt.zero_grad()
            loss = self.loss_func(y_pred,batch_y)
            loss.backward()
            opt.step()
            batch_loss.append(loss.item())

            #step2
            if self.args.func == 'FO':
                try:
                    batch_x, batch_y = next(loader_iter)
                except Exception:
                    loader_iter = iter(self.ldr_train)
                    batch_x, batch_y = next(loader_iter)
                batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)
                y_pred = net(batch_x)
                opt.zero_grad()
                loss = self.loss_func(y_pred,batch_y)
                loss.backward()
                for p1, p0 in zip(net.parameters(), temp_params):
                    p1.data = p0.data.clone()
                opt.step(beta=self.args.beta)
            else:
                try:
                    batch_x, batch_y = next(loader_iter)
                except Exception:
                    loader_iter = iter(self.ldr_train)
                    batch_x, batch_y = next(loader_iter)
                batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)

                temp1_params=copy.deepcopy(list(net.parameters()))
                y_pred = net(batch_x)
                opt.zero_grad()
                loss = self.loss_func(y_pred,batch_y)
                grad1 = torch.autograd.grad(loss, net.parameters(),retain_graph=True)
                opt.step(delta=self.args.delta,compute='add',grad=grad1)

                try:
                    batch_x, batch_y = next(loader_iter)
                except Exception:
                    loader_iter = iter(self.ldr_train)
                    batch_x, batch_y = next(loader_iter)
                batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)
                y_pred = net(batch_x)
                opt.zero_grad()
                loss = self.loss_func(y_pred, batch_y)
                grad2 = torch.autograd.grad(loss, net.parameters(), retain_graph=True)

                for p1,p0 in zip(net.parameters(),temp1_params):
                    p1.data = p0.data.clone()

                opt.zero_grad()
                opt.step(delta=self.args.delta,compute='sub',grad=grad1)

                y_pred = net(batch_x)
                opt.zero_grad()
                loss = self.loss_func(y_pred, batch_y)
                grad3 = torch.autograd.grad(loss, net.parameters(), retain_graph=True)

                #dw = (grad2-grad3)/(2*self.args.delta)
                dw = comp(grad2,grad3,self.args.delta)
                #grad = grad1-self.args.lr*dw
                grad = comp1(grad1,dw,self.args.lr)
                for p1, p0 in zip(net.parameters(), temp_params):
                    p1.data = p0.data.clone()
                opt.zero_grad()
                loss = self.loss_func(y_pred, batch_y)
                opt.step(beta=self.args.beta,grad=grad)

            batch_loss.append(loss.item())
            ep_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(ep_loss) / len(ep_loss)


if __name__=='__main__':
    args=args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    dataset_train, dataset_test,  dict_users=get_dataset1()

    net_glob = MlpMnist().to(args.device)
    print(net_glob)
    net_glob.train()

    w_glob = net_glob.state_dict()

    loss_train = []
    acc = []

    for k in range(args.epochs):
        loss_locals = []
        w_locals = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(arg=args, dateset=dataset_train, idx=dict_users[idx])
            w, loss = local.train(g_net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        acc_locals, _ = test_img(net_glob, dataset_test, args)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}, acc {:.3f}%'.format(k, loss_avg, acc_locals))
        loss_train.append(loss_avg)
        acc.append(acc_locals/100)

    # plot loss curve
    plt.figure(1)
    plt.plot(range(len(loss_train)), loss_train)
    plt.xlabel('round')
    plt.ylabel('train_loss')
    plt.title('FO(Mnist)')
    plt.savefig('./save/fed_{}_{}_{}_C_{}_iid_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    plt.figure(2)
    plt.plot(range(len(acc)), acc)
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.title('FO(Mnist)')
    plt.savefig('./save/fedacc_{}_{}_{}_C_{}_iid_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))