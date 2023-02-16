import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g,dataset,args):
    net_g.eval()

    test_loss=0
    correct=0
    data = DataLoader(dataset,args.test_bs)
    l=len(data)

    for idx ,(image,label) in enumerate(data):
        image,label = image.to(args.device),label.to(args.device)
        log_probs = net_g(image)
        test_loss+=F.cross_entropy(log_probs,label,reduction='sum').item()

        y_pred=log_probs.data.max(1,keepdim=True)[1]
        correct+=y_pred.eq(label.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data.dataset)
    accuracy = 100.00 * correct / len(data.dataset)

    #print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
    #        test_loss, correct, len(data.dataset), accuracy))

    return accuracy, test_loss