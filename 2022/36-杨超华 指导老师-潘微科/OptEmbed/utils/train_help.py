import logging
import sys
from models.modules_evo import *
from models.modules_super import *
from models.modules_retrain import *
from dataloader.tfloader import CriteoLoader, Avazuloader, KDD12loader
import torch
import pickle
import os

def get_dataloader(dataset, path):
    if dataset == 'Criteo':
        return CriteoLoader(path)
    elif dataset == 'Avazu':
        return Avazuloader(path)
    elif dataset == 'KDD12':
        return KDD12loader(path)

def get_stats(path):
    defaults_path = os.path.join(path + "/defaults.pkl")
    with open(defaults_path, 'rb') as fi:
        defaults = pickle.load(fi)
    if "criteo" in path:
        # return list(defaults.values())
        return [i+1 for i in list(defaults.values())]
    else:
        return [i+1 for i in list(defaults.values())] 

def get_supernet(opt):
    name = opt["model"]
    if name == "fm":
        model = FM_super(opt)
    elif name == "deepfm":
        model = DeepFM_super(opt)
    elif name == "fnn":
        model = FNN_super(opt)
    elif name == "ipnn":
        model = IPNN_super(opt)
    elif name == "dcn":
        model = DCN_super(opt)
    else:
        raise ValueError("Invalid model type: {}".format(name))
    return model

def get_evo(opt):
    name = opt["model"]
    if name == "fm":
        model = FM_evo(opt)
    elif name == "deepfm":
        model = DeepFM_evo(opt)
    elif name == "fnn":
        model = FNN_evo(opt)
    elif name == "ipnn":
        model = IPNN_evo(opt)
    elif name == "dcn":
        model = DCN_evo(opt)
    else:
        raise ValueError("Invalid model type: {}".format(name))
    return model

def get_retrain(opt):
    name = opt["model"]
    if name == "fm":
        model = FM_retrain(opt)
    elif name == "deepfm":
        model = DeepFM_retrain(opt)
    elif name == "fnn":
        model = FNN_retrain(opt)
    elif name == "ipnn":
        model = IPNN_retrain(opt)
    elif name == "dcn":
        model = DCN_retrain(opt)
    else:
        raise ValueError("Invalid model type: {}".format(name))
    return model

def get_optimizer(network, opt):
    threshold_params, network_params, embedding_params = [], [], []
    threshold_names, network_names, embedding_names = [], [], []
    for name, param in network.named_parameters():
        if name == "threshold":
            threshold_params.append(param)
            threshold_names.append(name)
        elif name == "embedding":
            embedding_params.append(param)
            embedding_names.append(name)
        else:
            network_params.append(param)
            network_names.append(name)
    
    print("threshold_names:", threshold_names)
    print("_"*30)
    print("embedding_names:", embedding_names)
    print("_"*30)
    print("network_names:", network_names)
    print("_"*30)

    threshold_group = {
        'params': threshold_params,
        'lr': opt['threshold_lr']
    }
    threshold_optimizer = torch.optim.SGD([threshold_group])

    embedding_group = {
        'params': embedding_params,
        'weight_decay': opt['wd'],
        'lr': opt['lr']
    }
    network_group = {
        'params': network_params,
        'weight_decay': opt['wd'],
        'lr': opt['lr']
    }
    if opt['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD([network_group, embedding_group])
    elif opt['optimizer'] == 'adam':
        optimizer = torch.optim.Adam([network_group, embedding_group])
    else:
        print("Optimizer not supported.")
        sys.exit(-1)

    return threshold_optimizer, optimizer

def get_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)

def get_log(name=""):
    FORMATTER = logging.Formatter(fmt="[{asctime}]:{message}", style= '{')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(FORMATTER)
    logger.addHandler(ch)
    return logger
