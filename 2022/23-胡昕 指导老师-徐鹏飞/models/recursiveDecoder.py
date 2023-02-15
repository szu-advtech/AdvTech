#!/usr/bin/env python
# encoding: utf-8
# author: Hu Xin
# email: qzlyhx@gmail.com
# time: 2022.8.10
# description:
#   This file contains model-decoder structures and method 



import torch
import torch.nn.functional as F
import torch.nn as nn
import models.tree as tree
from utils.config import getDevice
from utils.positionMapper import PositionMapper


# The class specifies the MLP structure of decoder
class DecoderUnit(nn.Module):
    def __init__(self, config:object) -> None:
        super(DecoderUnit, self).__init__()
        device = getDevice()
        #MLP with 1 hidden layer
        self.fc1 = nn.Linear(config.decoderInputSize, config.decoderHiddenLayerSize, device=device)
        self.fc2 = nn.Linear(config.decoderHiddenLayerSize, config.decoderOutputSize, device=device)

        
    def initParameters(self, initFunc):
        initFunc(self.fc1.weight, std=0.03)
        initFunc(self.fc2.weight, std=0.03)
        initFunc(self.fc1.bias, std=0.03)
        initFunc(self.fc2.bias, std=0.03)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


# this unit convert fixed length Leaf features back to Actomic unit features
class FeatureMapBackUnit(nn.Module):
    def __init__(self, config:object) -> None:
        super(FeatureMapBackUnit, self).__init__()
        device = getDevice()
        # SLP to convert Leaf node to features
        self.fc = nn.Linear(config.nodeFeatureSize, config.atomicUnitSize, device=device)
    
    def initParameters(self, initFunc):
        initFunc(self.fc.weight, std=0.03)
        initFunc(self.fc.bias, std=0.03)

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        return x


# This class is a classifier unit that judges whether a node cis a Leaf or not
# output x > 0.5 -> is a node else leaf
class ClassifierUnit(nn.Module):
    def __init__(self, config:object) -> None:
        super(ClassifierUnit, self).__init__()
        device = getDevice()
        #MLP
        self.fc1 = nn.Linear(config.nodeFeatureSize, config.classifierHiddenLayerSize, device=device)
        self.fc2 = nn.Linear(config.classifierHiddenLayerSize, 7, device=device) #7 classifier outputs 

    def initParameters(self, initFunc):
        initFunc(self.fc1.weight, std=0.03)
        initFunc(self.fc2.weight, std=0.03)
        initFunc(self.fc1.bias, std=0.03)
        initFunc(self.fc2.bias, std=0.03)

    def forward(self, x):
        # the gradient of classifer 'should' be fed back to x otherwise the classifier would not work well
        # x = x.clone().detach()
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        return x

class LeafClassifierUnit(nn.Module):
    def __init__(self, config:object) -> None:
        super(LeafClassifierUnit, self).__init__()
        device = getDevice()
        #MLP
        self.fc1 = nn.Linear(config.nodeFeatureSize, config.classifierHiddenLayerSize, device=device)
        self.fc2 = nn.Linear(config.classifierHiddenLayerSize, 1, device=device) # 1 classifier outputs 

    def initParameters(self, initFunc):
        initFunc(self.fc1.weight, std=0.03)
        initFunc(self.fc2.weight, std=0.03)
        initFunc(self.fc1.bias, std=0.03)
        initFunc(self.fc2.bias, std=0.03)

    def forward(self, x):
        # the gradient of classifer 'should' be fed back to x otherwise the classifier would not work well
        # x = x.clone().detach()
        x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x


class RecursiveDecoder():
    def __init__(self, config:object) -> None:
        # init sub units
        self.decoderUnit = dict()
        for type in PositionMapper.getTypes():
            self.decoderUnit[type] = DecoderUnit(config)

        self.featureMapBackUnit = FeatureMapBackUnit(config)
        self.classifierUnit = ClassifierUnit(config)
        self.leafClassifierUnit = LeafClassifierUnit(config)
        # some other configurations
        self.maxDepth = config.maxTreeDepth
        self.config = config

    def getParameters(self):
        paraList = []
        paraList.append({"params":self.featureMapBackUnit.parameters()}) 
        paraList.append({"params":self.classifierUnit.parameters()}) 
        paraList.append({"params":self.leafClassifierUnit.parameters()}) 
        for type in PositionMapper.getTypes():
            paraList.append({"params":self.decoderUnit[type].parameters()})  
        return paraList

    def decodeToTree(self, feature, gtTree=None) -> tree.Tree:
        rootNode = tree.Node(nodeData=feature)
        if gtTree == None:
            self.decode(rootNode, self.maxDepth)
        else:
            self.decodeGT(rootNode, gtTree.rootNode)
        targetTree = tree.Tree(rootNode=rootNode)
        return targetTree
    
    def decodeGT(self, node:tree.Node, gtNode:tree.Node):
        rootNode = gtNode
        config = self.config
        targetNode = node
        while rootNode.child1 != None:
            type = PositionMapper.onehotToPosition(rootNode.nodeRelativePositionData.squeeze())
            targetNode.nodeNotLeafProbability =torch.cat( [self.leafClassifierUnit.forward(targetNode.nodeData), self.classifierUnit.forward(targetNode.nodeData)], dim=1)

            decodedFeatures = self.decoderUnit[type](targetNode.nodeData)
            child1Data = decodedFeatures[:,0:config.nodeFeatureSize]
            child2Data = decodedFeatures[:,config.nodeFeatureSize:(2*config.nodeFeatureSize)]
            detailedRelativePosition = decodedFeatures[:,(2*config.nodeFeatureSize):]
            targetNode.setNodeDetailedRelativePositionData(detailedRelativePosition)
            targetNode.setNodeRelativePositionData(rootNode.nodeRelativePositionData)
            # create 2 child node structure
            childNode1 = tree.Node(nodeData=child1Data)
            childNode2 = tree.Node(nodeData=child2Data)
            childNode2.setLeafAtomicUnitData(self.featureMapBackUnit(childNode2.nodeData))
            childNode2.nodeNotLeafProbability = torch.cat( [self.leafClassifierUnit.forward(childNode2.nodeData), self.classifierUnit.forward(childNode2.nodeData)], dim=1)
            targetNode.setChild1(childNode1)
            targetNode.setChild2(childNode2)
            targetNode = targetNode.child1
            rootNode = rootNode.child1
        # finally rootNode is the last leaf
        targetNode.setLeafAtomicUnitData(self.featureMapBackUnit(targetNode.nodeData))
        targetNode.nodeNotLeafProbability = torch.cat( [self.leafClassifierUnit.forward(targetNode.nodeData), self.classifierUnit.forward(targetNode.nodeData)], dim=1)

    # recursive function
    def decode(self, node:tree.Node, depthNow:int):
        # judge whether Leaf node
        classfierOut = torch.cat( [self.leafClassifierUnit.forward(node.nodeData), self.classifierUnit.forward(node.nodeData)], dim=1)
        # print(classfierOut)
        isLeaf = True
        if classfierOut[:, 0:1].sum() > 0.5:
            isLeaf = False
        node.nodeNotLeafProbability = classfierOut

        # to force recursion stop at depth 0
        if depthNow < 2:
            isLeaf = True

        config = self.config
        
        targetNode = node
        # Dealing with different Leaf conditions
        if isLeaf:
            node.setLeafAtomicUnitData(self.featureMapBackUnit(node.nodeData))
            targetNode = node
        else:

            # node is not a Leaf, first decode it
            type = PositionMapper.onehotToPosition(classfierOut[:, 1:8].squeeze())
            decodedFeatures = self.decoderUnit[type](node.nodeData)
            child1Data = decodedFeatures[:,0:config.nodeFeatureSize]
            child2Data = decodedFeatures[:,config.nodeFeatureSize:(2*config.nodeFeatureSize)]
            detailedRelativePosition = decodedFeatures[:,(2*config.nodeFeatureSize):]
            # create 2 child node structure
            childNode1 = tree.Node(nodeData=child1Data)
            childNode2 = tree.Node(nodeData=child2Data)
            childNode2.setLeafAtomicUnitData(self.featureMapBackUnit(childNode2.nodeData))
            childNode2.nodeNotLeafProbability = torch.cat( [self.leafClassifierUnit.forward(childNode2.nodeData), self.classifierUnit.forward(childNode2.nodeData)], dim=1)
            node.setNodeRelativePositionData(classfierOut[:, 1:8])
            node.setNodeDetailedRelativePositionData(detailedRelativePosition)
            node.setChild1(childNode1)
            node.setChild2(childNode2)
            # enter recursive
        
            self.decode(node.child1, depthNow - 1)
            # self.decode(node.child2, depthNow - 1)
            targetNode = node
        
        # return decoded mode
        return targetNode
