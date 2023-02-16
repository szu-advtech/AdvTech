import torch
from torch.utils.data import DataLoader
from ScaffoldOptimizer import ScaffoldOptimizer
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from functools import reduce
import copy
def LocalTrain(args, data, net, server_controls, client_controls):
    train = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)
    #optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer = ScaffoldOptimizer(net.parameters(), lr=args.lr, weight_decay = 0)
    Loss = []
    net.train()
    for iter in range(args.local_ep):
        for images, labels in train:
            images = images.to(args.device)
            labels = labels.to(args.device)
            net.zero_grad()
            predicts = net(images)
            loss = args.cost(predicts, labels)
            Loss.append(loss.item())
            loss.backward()
            optimizer.step(server_controls, client_controls)

    #next_client_controls = [copy.deepcopy(params.grad.data) for params in net.parameters()]

    net.zero_grad()
    for images, labels in DataLoader(dataset=data, batch_size=args.test_batch, shuffle=True):
        images = images.to(args.device)
        labels = labels.to(args.device)
        predicts = net(images)
        loss = args.cost(predicts, labels)
        loss.backward()

    next_client_controls = [copy.deepcopy(params.grad.data)/(len(data)/args.test_batch) for params in net.parameters()]
    return net, sum(Loss)/len(Loss), next_client_controls