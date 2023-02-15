import torch
from torch import nn
from torchvision import datasets, transforms, models
import Clients
import Model as m
import random
import numpy as np
import ssl
import os
import math
import scipy.stats as st

ssl._create_default_https_context = ssl._create_unverified_context
myseed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(myseed)
os.environ['PYTHONHASHSEED'] = str(myseed)
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

client_num = 100
select_num = 10
batch_size = 20

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # R,G,B每层的归一化用到的均值和方差
])

if __name__ == '__main__':
    tmp_model = torch.load(
            "../init_conv_model.pkl")  # m.ConvMixer(256, 8, patch_size=2, kernel_size=5, n_classes=10)
    model = torch.load(
            "../init_conv_model.pkl")  # m.ConvMixer(256, 8, patch_size=2, kernel_size=5, n_classes=10)
    d = 0
    for p in model.parameters():
        d += p.numel()
    print(f'{d:,} training parameters.')

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../../CIFAR-10', train=False, download=True,
                         transform=transform_test),
        batch_size=batch_size, shuffle=False
    )
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../../CIFAR-10', train=True, download=True,
                         transform=transform_test),
        batch_size=batch_size, shuffle=True
    )

    # 确定客户端的数量，从而进行数据的划分
    clients = Clients.Client(client_num)

    communication_time = 0  # 记录当前第几轮通信

    m = torch.zeros(d)
    v = torch.zeros(d)
    vhat = torch.zeros(d)
    epsilon = torch.ones(d) * 0.001

    beta1 = 0.9
    beta2 = 0.99
    eta = 1

    # 一个迭代回合代表一次全局通信
    while True:
        if communication_time % 5 == 0:
            acc = 0
            train_loss = 0.
            correct = 0.
            total = 0.
            loss_fn = torch.nn.CrossEntropyLoss()
            model = model.to(device)

            with torch.no_grad():
                model.train()
                for (data, target) in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                acc = correct / total * 100.
                print("Communication Time is ", communication_time)
                print("Accuracy: {}".format(acc))

                # for (data, target) in train_loader:
                #     data = data.to(device)
                #     target = target.to(device)
                #     output = model(data)
                #     train_loss += loss_fn(output, target)
                # train_loss /= len(train_loader)
                # print("Train loss: {}".format(train_loss))
            model = model.to('cpu')
            if communication_time >= 200:
                print("Communication Time is ", communication_time)
                break

        select_clients = random.sample(list(range(client_num)), select_num)

        model_gra = torch.zeros(d)  # 全局梯度

        for j in select_clients:
            print("the select client num is {}".format(j))
            clients.set_model(model)

            clients.train(j, communication_time)

            local_model_gra, num_batches_tracked = clients.get_model_gradient(j)  # 客户端的时间,同时包括上传\重传请求

            model_gra += 1 / select_num * local_model_gra

        # m = beta1 * m + (1 - beta1) * model_gra
        # v = beta2 * v + (1 - beta2) * (model_gra ** 2)
        # vhat = torch.max(vhat, torch.max(v, epsilon))
        #
        # model_gra = eta * m / torch.sqrt(vhat)
        # model_gra = eta * m

        idx = 0
        for p in model.parameters():
            size = p.numel()
            p.data = p.data + model_gra[idx:idx + size].reshape(
                p.shape).data
            # else:
            #     m[idx:idx + size] = beta1 * m[idx:idx + size] + (1 - beta1) * model_gra[idx:idx + size]
            #     v[idx:idx + size] = beta2 * v[idx:idx + size] + (1 - beta2) * (model_gra[idx:idx + size] ** 2)
            #     vhat[idx:idx + size] = torch.max(vhat[idx:idx + size],
            #                                      torch.max(v[idx:idx + size], epsilon[idx:idx + size]))
            #
            #     model_gra[idx:idx + size] = eta * m[idx:idx + size] / torch.sqrt(vhat[idx:idx + size])
            #
            #     new_model[p].data = new_model[p].data + model_gra[idx:idx + size].reshape(new_model[p].shape).data
            idx += size


        communication_time += 1
        print("Communication Time is ", communication_time)

        # if communication_time % 50 == 0:
        #     clients.adjust_learning_rate()

