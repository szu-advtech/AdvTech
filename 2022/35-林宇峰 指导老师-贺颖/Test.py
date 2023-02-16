import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return float(rights)/len(labels)

def test(args, data, net):
    test = DataLoader(dataset = data, batch_size = args.test_batch, shuffle=True)
    Loss = []
    Accuracy = []
    net.eval()
    for images, labels in test:
        images = images.to(args.device)
        labels = labels.to(args.device)
        predicts = net(images)
        loss = args.cost(predicts, labels)
        Loss.append(loss.item())
        acc = accuracy(predicts, labels)
        Accuracy.append(acc)
    return sum(Accuracy)/len(Accuracy),sum(Loss)/len(Loss)