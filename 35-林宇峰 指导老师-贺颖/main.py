import copy
import net
import torch
import Fed
import Update
import Test
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
import sampling

#程序预设值
class Args(object):
    epochs = 1000
    num_users = 10
    dataset = 'Mnist'#'Cifar10'
    lr = 0.01
    local_ep = 10

    batch_size = 1000
    test_batch = 1000
    frac = 0.5
    iid = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cost = nn.CrossEntropyLoss()


#************************************************
args = Args()

if args.dataset == 'Mnist':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    if args.iid:
        local_trainset = sampling.mnist_iid(trainset,args.num_users)
    else:
        local_trainset = sampling.mnist_noniid(trainset, args.num_users)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    globe_net = net.Mnist_net().to(args.device)

if args.dataset == 'Cifar10':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    if args.iid:
        local_trainset = sampling.cifar_iid(trainset,args.num_users)
    else:
        exit('error: only consider IID setting on cifar10')
        #local_trainset = sampling.mnist_noniid(trainset, args.num_users)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    globe_net = net.Cifar10_net().to(args.device)

def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return float(rights)/len(labels)
def average(Grad):
    answer = torch.zeros_like(Grad[0])
    for grad in Grad:
        answer += grad
    return answer/len(Grad)

loss_train = []
cost = nn.CrossEntropyLoss()
server_controls = None
client_controls = [None for i in range(args.num_users)]
for epoch in range(args.epochs):

    local_net = {customer: copy.deepcopy(globe_net).to(args.device) for customer in range(args.num_users)}
    loss_locals = []
    m = max(int(args.frac*args.num_users),1)
    choice = np.random.choice(range(args.num_users), m, replace = False)
    for customer in choice:
        local_net[customer], loss, client_controls[customer] = Update.LocalTrain(args, local_trainset[customer], local_net[customer],server_controls, client_controls[customer])
        loss_locals.append(copy.deepcopy(loss))
    globe_net.load_state_dict(Fed.FedAvg([local_net[i].state_dict() for i in range(len(local_net))]))#利用选取的局部模型对全局模型进行聚合更新
    server_controls = [average(grad) for grad in zip(*[client_controls[customer] for customer in choice])]
    loss_avg = sum(loss_locals)/len(loss_locals)
    print('epoch:',epoch)
    print('    local Average loss:', loss_avg)

    acc, loss = Test.test(args, trainset, globe_net)
    print('    serve accuracy:',acc)
    print('    serve loss', loss)
    loss_train.append(loss)
plt.figure()
plt.plot(range(len(loss_train)), loss_train)
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.savefig('./SCAFFOLD.png')


