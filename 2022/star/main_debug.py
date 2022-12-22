import requests
import torch

from main_node import getNetGlob
from models.Nets import CNN
from utils.dataset import test_img
from utils.models import hexToStateDict
from utils.options import args_debug_parser
from torchvision import datasets, transforms

args = args_debug_parser()
def getTrainInfo():
    resp = requests.post(args.rpc_url, json={
        "jsonrpc": "2.0",
        "method": "eth_getTrainInfo",
        "params": [],
        "id": 1
    })
    data = resp.json()
    return data['result']

def getNetGlobEmpty():
    trainInfo = getTrainInfo()
    if trainInfo['model_name'] == "cnn":
        model = CNN(num_channels=trainInfo['num_channels'], num_classes=trainInfo['num_classes'])
    return model


def getModelByBlockNumber(blockNumber):
    model = getNetGlobEmpty()
    data = requests.post(args.rpc_url, json={
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [
            hex(blockNumber),
            True
        ],
        "id": 1
    }).json()
    result = data['result']
    try:
        stateDict = hexToStateDict(result['extraData'][2:])
        model.load_state_dict(stateDict)
    except:
        print('init')

    return model


if __name__ == '__main__':
    device = torch.device('mps'.format(args.gpu) if torch.has_mps and args.gpu != -1 else 'cpu')

    if args.dataset == "mnist":
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == "cifar":
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    else:
        exit("don't support dataset")
    for i in range(103):
        net_glob = getModelByBlockNumber(i).to(device=device)
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args.epoch, args.gpu)


