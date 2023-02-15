import csv

import torch
from torchvision import datasets, transforms  # 导入数据集与数据预处理的方法
import utils
import KMT
import numpy as np
import random

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = datasets.FashionMNIST('dataset/', download=False, train=True, transform=transform)

test_set = datasets.FashionMNIST('dataset/', download=False, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=200, shuffle=True)

client_number = 10
Q_client = 4
rounds = 45
pre_train_time = 5
ambiguous_number = 200
heterogeneity = 8
client_sample_number = 6000

indexes = utils.non_iid_divide(train_set, client_number, heterogeneity, client_sample_number)
clients = {}
for i in range(client_number):
    index = np.array(list(indexes[i]))
    clients[i] = KMT.KMT(train_set, index, test_loader)

# scheduler = utils.schedule(client_number, Q_client, rounds)
scheduler = np.zeros([2*rounds, Q_client])
accuracy = np.zeros([rounds, client_number])

for i in range(client_number):
    print("client " + str(i) + " is pre-training.......")
    for j in range(pre_train_time):
        clients[i].train()
        clients[i].test()
    print("client " + str(i) + "'s accuracy is " + str(clients[i].test_accuracy[len(clients[i].test_accuracy)-1]))

for i in range(rounds):
    scheduler[i * 2 + 1] = random.sample(range(client_number), Q_client)
    p = 0
    for j in scheduler[i * 2 + 1]:
        ambiguous_array = clients[j].memory.get_ambiguous(ambiguous_number)
        ambiguous_images, ambiguous_labels = clients[j].get_ambiguous_data(ambiguous_array)
        min_loss = 1000
        choose_client = -1
        for k in range(client_number):
            # if j != k and not list(scheduler[i * 2]).__contains__(k):
            if j != k:
                temp_loss = clients[k].test_data(ambiguous_images, ambiguous_labels)
                if temp_loss < min_loss:
                    min_loss = temp_loss
                    choose_client = k
        scheduler[i * 2][p] = choose_client
        p += 1
    print("_______________________")
    print("round " + str(i + 1))
    for j in set(scheduler[i * 2, :]):
        print("client " + str(int(j)) + " is training.......")
        clients[j].train()
    for k in range(Q_client):
        print("client " + str(scheduler[i * 2 + 1, k]) + " is training with the help of " + str(
            scheduler[i * 2, k]) + ".......")
        client1 = clients[scheduler[i * 2, k]]
        client2 = clients[scheduler[i * 2 + 1, k]]
        client1.save_model()
        for images, labels in client2.train_loader:
            batch = 0
            other_opinion = torch.softmax(client1.model(images), dim=1).data
            my_opinion = torch.softmax(client2.model(images), dim=1).data
            client1.distillation_train(images, labels, my_opinion)
            client2.memory.add()
            client2.memory.add_number(batch, client2.distillation_train(images, labels, other_opinion))
        client1.save_my_model()
        client2.load_other_model()
        client1.load_model()
    for m in range(client_number):
        accuracy[i, m] = clients[m].test().item()
        print("client "+str(m)+"'s accuracy is "+str(accuracy[i, m]))


csv_file = open('result.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(csv_file)
for i in range(client_number):
    writer.writerow(str(i))
    writer.writerow(clients[i].train_loss)
    writer.writerow(clients[i].test_loss)
    writer.writerow(clients[i].test_accuracy)
csv_file.close()
