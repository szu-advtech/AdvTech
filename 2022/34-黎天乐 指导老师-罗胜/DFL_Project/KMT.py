from torch.utils.data import DataLoader
import dnn_model
import torch
from torch import nn
import numpy as np
from memory import memory


class KMT:
    learning_rate = 0.01
    momentum = 0.5
    batch_size = 200
    batch_number = 30
    load_model = None
    memory = None
    train_loader = None
    validate_loader = None
    test_loader = None
    train_loss = []
    test_loss = []
    test_accuracy = []

    def __init__(self, dataset, index, test_loader):
        self.load_train_set(dataset, index)
        self.test_loader = test_loader
        self.model = dnn_model.CNNFashion_Mnist(nn.Module)
        self.criterion = nn.CrossEntropyLoss()
        self.distillation_criterion = nn.KLDivLoss(reduction='batchmean')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.memory = memory(self.batch_number, self.batch_size)
        self.train_loss = []
        self.test_loss = []
        self.test_accuracy = []

    def load_train_set(self, dataset, index):
        train_dataset = torch.utils.data.Subset(dataset, index.astype(int))
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)

    def train(self):
        running_loss = 0
        batch_number = 0
        self.memory.add()
        for images, labels in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)

            log_ps = torch.log_softmax(output, dim=1)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            equals = equals.numpy().squeeze().astype(int).transpose()
            self.memory.add_number(batch_number, equals)

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            batch_number += 1
        self.train_loss.append(running_loss / len(self.train_loader))

    def distillation_train(self, images, labels, other_soft_decision):
        self.optimizer.zero_grad()
        output = self.model(images)
        loss1 = self.criterion(output, labels)
        loss2 = self.distillation_criterion(torch.log(torch.softmax(output, dim=1)), other_soft_decision)
        loss = loss1 + loss2

        log_ps = torch.log_softmax(output, dim=1)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        equals = equals.numpy().squeeze().astype(int).transpose()

        loss.backward()
        self.optimizer.step()
        return equals

    def test(self):
        accuracy = 0
        loss = 0
        for images, labels in self.test_loader:
            output = self.model(images)
            log_ps = torch.log_softmax(output, dim=1)
            ps = torch.exp(log_ps)
            loss += self.criterion(log_ps, labels)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        self.test_loss.append((loss / len(self.test_loader)).item())
        self.test_accuracy.append((accuracy / len(self.test_loader)).item())
        return accuracy / len(self.test_loader)

    def save_model(self):
        torch.save(self.model.state_dict(), "temp_model.pt")

    def save_my_model(self):
        torch.save(self.model.state_dict(), "my_model.pt")

    def load_model(self):
        self.model.load_state_dict(torch.load("temp_model.pt"))

    def load_other_model(self):
        self.model.load_state_dict(torch.load("my_model.pt"))

    def get_ambiguous_data(self, array):
        np.sort(array)
        batch_number = 0
        ambiguous_images = []
        ambiguous_labels = []
        for images, labels in self.train_loader:
            for no in array:
                if int(no / self.batch_size) == batch_number:
                    column = np.mod(no, self.batch_size)
                    ambiguous_images.append(images[column])
                    ambiguous_labels.append(labels[column])
            batch_number += 1
        ambiguous_images = torch.stack(ambiguous_images)
        ambiguous_labels = torch.stack(ambiguous_labels)
        return ambiguous_images, ambiguous_labels

    def test_data(self, images, labels):
        loss = 0
        output = self.model(images)
        log_ps = torch.log_softmax(output, dim=1)
        loss += self.criterion(log_ps, labels)
        return loss

    def test_data_accuracy(self, images, labels):
        output = self.model(images)
        log_ps = torch.log_softmax(output, dim=1)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        return accuracy
