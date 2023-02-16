#!/usr/bin/env python
# encoding: utf-8
# author: Hu Xin
# email: qzlyhx@gmail.com
# time: 2022.8.9
# description:
#   This file contains model-encoder structures and method 

import torch
import torch.nn.functional as F
import torch.nn as nn
import models.tree as tree
from utils.config import getDevice
from utils.positionMapper import PositionMapper


# The class specifies the MLP structure of encoder
class EncoderUnit(nn.Module):
    def __init__(self, config:object) -> None:
        super(EncoderUnit, self).__init__()
        device = getDevice()
        #MLP with 1 hidden layer
        self.fc1 = nn.Linear(config.encoderInputSize, config.encoderHiddenLayerSize, device=device)
        self.fc2 = nn.Linear(config.encoderHiddenLayerSize, config.encoderOutputSize, device=device)
        
    def initParameters(self, initFunc):
        initFunc(self.fc1.weight, std=0.03)
        initFunc(self.fc2.weight, std=0.03)
        initFunc(self.fc1.bias, std=0.03)
        initFunc(self.fc2.bias, std=0.03)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
    
    


# This class convert original atomic unit features, including the loction of the bounding box and the type of the unit,
# TO a fixed length feature that could be fed into RvNN
class LeafFeaturizeUnit(nn.Module):
    def __init__(self, config:object) -> None:
        super(LeafFeaturizeUnit, self).__init__()
        device = getDevice()
        # SLP to convert Leaf node to features
        self.fc = nn.Linear(config.atomicUnitSize, config.nodeFeatureSize, device=device)
    
    def initParameters(self, initFunc):
        initFunc(self.fc.weight, std=0.03)
        initFunc(self.fc.bias, std=0.03)

    def forward(self, x):
        # print(x.is_cuda)
        x = torch.tanh(self.fc(x))
        return x


# this class encodes a document tree to a fixed length feature
class RecursiveEncoder():
    def __init__(self, config:object) -> None:
        self.leafUnit = LeafFeaturizeUnit(config)
        self.encoderUnit = dict()
        for type in PositionMapper.getTypes():
            self.encoderUnit[type] = EncoderUnit(config)
    
    # this function returns all parameters in encoder
    def getPatameters(self):
        paraList = []
        paraList.append({"params":self.leafUnit.parameters()})
        for type in PositionMapper.getTypes():
            paraList.append({"params":self.encoderUnit[type].parameters()}) 
        return paraList

    def encodeATree(self, treeData:tree.Tree):
        rootNode = treeData.rootNode
        encodedFeature = self.encode(rootNode)
        return encodedFeature

    def encode(self, node:tree.Node):
        isLeaf = False
        if (node.child1 == None and node.child2 == None):
            isLeaf = True
        
        targetFeature = None
        if isLeaf:
            targetFeature = self.leafUnit(node.leafAtomicUnitData)
            node.nodeData = targetFeature
        else:
            #concatenate datas
            child1Feature = self.encode(node.child1)
            child2Feature = self.encode(node.child2)
            detailedRelativePositionFeature = node.nodeDetailedRelativePositionData
            x = torch.cat([child1Feature, child2Feature, detailedRelativePositionFeature], dim=1)
            # now get relativePosition type and apply corresponding encoder
            type = PositionMapper.onehotToPosition(node.nodeRelativePositionData.squeeze())
            targetFeature = self.encoderUnit[type](x)
            node.nodeData = targetFeature
        
        return targetFeature
    