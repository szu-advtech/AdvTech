#!/usr/bin/env python

import logging

import torch


logger = logging.getLogger(__name__)

def freeze_(model, layer='', _grad=False):
    """ Freeze one or more layers
    """
    for name, param in model.named_parameters():
        if layer in name:  # '' matches all
            param.requires_grad_(_grad)

def unfreeze_(model, layer=''):
    freeze_(model, layer, _grad=True)

def stripClassifier(model):
    """ Return model up until the classifier
    """
    module_list = list()
    for module in model.children():
        if module is not model.classifier:
            module_list.append(module)
        else:
            break
 
    return torch.nn.Sequential(*module_list)

def loadFromHub(config):
    parameters = list()
    named_parameters = dict()

    for param in config:
        if '=' not in param:
            parameters.append(param)
            continue

        key, value = param.split('=')
        named_parameters[key.strip()] = value.strip()
        print(*parameters)
        print(**named_parameters)
    return torch.hub.load(*parameters, **named_parameters)

def inferOutputDim(model):
    """ Assume that the output dimension of last linear or Conv layer
        equals that of the entire model.
    """
    out_dim = -1
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Linear):
            out_dim = module.out_features
            break
        if isinstance(module, torch.nn.Conv2d):
            out_dim = module.out_channels
            break

    return out_dim

def getPadSymbol(tokenizer_cfg):
    tokenizer = loadFromHub(tokenizer_cfg['config'])
    pad_token = tokenizer_cfg['pad_token']

    return tokenizer.encode(pad_token, add_special_tokens=False)[0]

def torch_intersect1d(ta, tb):
    return ta[(ta.view(1, -1) == tb.view(-1, 1)).any(dim=0)]
