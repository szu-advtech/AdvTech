#!/usr/bin/env python
# encoding: utf-8
# author: Hu Xin
# email: qzlyhx@gmail.com
# time: 2022.8.10
# description:
#   This file calculates leaf-level reconstruction loss, relative-position reconstruction loss and  categorical cross-entropy loss

from xml.etree.ElementTree import tostring
import torch
from models.tree import Tree

from models.tree import Node
import os

from utils.config import getDevice
from utils.positionMapper import PositionMapper
from utils.typeMapper import TypeMapper

temp = None
def calculateLossBetweenTrees(targetTree:Tree, gtTree:Tree):
    # calculate loss recursively
    losses = __calculateLoss(targetTree.rootNode, gtTree.rootNode)
    #返回3个loss
    leafRecLoss, relativePosLoss, ceLoss, isLeafLoss = 0, 0, 0, 0
    if losses[0][0] != 0:
        # leafRecLoss = losses[0][1]/losses[0][0]
        leafRecLoss = losses[0][1]
    if losses[1][0] != 0:
        # relativePosLoss = losses[1][1]/losses[1][0]
        relativePosLoss = losses[1][1]
    if losses[2][0] != 0:
        # ceLoss = losses[2][1]/losses[2][0]
        ceLoss = losses[2][1]
    if losses[3][0] != 0:
        # isLeafLoss = losses[3][1]/losses[3][0]
        isLeafLoss = losses[3][1]
    # if losses[3][0] == 1:
    #     print("aaa", temp)
    return leafRecLoss, relativePosLoss, ceLoss, isLeafLoss


# returns (num of added Leaf loss, Lleaf), (num of added pos loss, Lpos), (num of added celoss, Lce), (num of isLeafLoss, Lisleaf)
def __calculateLoss(node:Node, nodeGt:Node):
    if node == None or nodeGt == None:
        return (0, 0), (0, 0), (0, 0), (0, 0)
    
    ceLossFunc = torch.nn.CrossEntropyLoss()
    mseLossFunc = torch.nn.MSELoss()
    l1LossFunc = torch.nn.L1Loss()
    BCELossFunc = torch.nn.BCELoss()
    device = getDevice()

    
    # to train classifier in decoder
    leafRange = 0
    if nodeGt.child1 != None or nodeGt.child1 != None:
        leafRange = 1
    isLeafFeat = torch.tensor(leafRange, dtype=torch.float32, device=device).expand([1])
    positionFeat = None

    if (nodeGt.nodeRelativePositionData == None):
        # This is a leaf,set all to 0
        positionFeat = torch.tensor(0, dtype=torch.float32, device=device).expand([7])
    else:
        positionFeat = nodeGt.nodeRelativePositionData.squeeze()

    classifierGTFeat = torch.cat([isLeafFeat, positionFeat], dim=0)
    

    # ceLoss = ceLossFunc(softmax1.unsqueeze(dim=0), softmaxGt.unsqueeze(dim=0))
    ceLoss = ceLossFunc( node.nodeNotLeafProbability[:, 1:8], classifierGTFeat[1:8].unsqueeze(dim=0))

    #isLeafLoss
    # global temp
    # temp = node.nodeNotLeafProbability.squeeze()[0:7], classifierGTFeat[0:7]
    # print(node.nodeNotLeafProbability.squeeze(), classifierGTFeat)
    isLeafLoss = BCELossFunc( node.nodeNotLeafProbability[:, 0:1], classifierGTFeat[0:1].unsqueeze(dim=0))

    leafLoss = 0
    # node and nodeGt are both leafs
    if nodeGt.child1 == None and node.child1 == None:
        leafLoss = mseLossFunc(node.leafAtomicUnitData[:, 0:2], nodeGt.leafAtomicUnitData[:, 0:2])
        leafLoss = leafLoss +  ceLossFunc(node.leafAtomicUnitData[:, 2:2 + TypeMapper.getTypeNumbers()], nodeGt.leafAtomicUnitData[:, 2:2 + TypeMapper.getTypeNumbers()])
        if leafLoss > 100:
            print("unexpected Leaf Loss - " + str(leafLoss))
            print(node.leafAtomicUnitData, nodeGt.leafAtomicUnitData)
            os.system("pause")
        # no ce Loss
        return (1, leafLoss), (0, 0), (0, 0), (1, isLeafLoss)

    if nodeGt.child1 != None and node.child1 != None:
        # ###check whether classifier outputs the right result ###
        # type = PositionMapper.onehotToPosition(node.nodeNotLeafProbability.squeeze()[7:14])
        # typegt = PositionMapper.onehotToPosition(classifierGTFeat[7:14])
        # if type != typegt:
        #     # print(type, typegt)
        #     # print(node.nodeNotLeafProbability.squeeze()[7:14], classifierGTFeat[7:14])
        #     return (0, 0), (0, 0), (1, ceLoss), (1, isLeafLoss)

        #both are not Leaf
        child1Loss = __calculateLoss(node.child1, nodeGt.child1)
        child2Loss = __calculateLoss(node.child2, nodeGt.child2)
        # calculate relative-position reconstruction loss
        posLoss = mseLossFunc(node.nodeDetailedRelativePositionData[:, 0:2], nodeGt.nodeDetailedRelativePositionData[:, 0:2])
        posLoss = posLoss + ceLossFunc(node.nodeDetailedRelativePositionData[:, 2:6], nodeGt.nodeDetailedRelativePositionData[:, 2:6])
        posLoss = posLoss + ceLossFunc(node.nodeDetailedRelativePositionData[:, 6:10], nodeGt.nodeDetailedRelativePositionData[:, 6:10])
        # return aggregated losses
        return  (child1Loss[0][0] + child2Loss[0][0], child1Loss[0][1] + child2Loss[0][1]), \
                (child1Loss[1][0] + child2Loss[1][0] + 1, child1Loss[1][1] + child2Loss[1][1] + posLoss), \
                (child1Loss[2][0] + child2Loss[2][0] + 1, child1Loss[2][1] + child2Loss[2][1] + ceLoss),\
                (child1Loss[3][0] + child2Loss[3][0] + 1, child1Loss[3][1] + child2Loss[3][1] + isLeafLoss)

    # in case that node type not matched
    return (0, 0), (0, 0), (1, ceLoss), (1, isLeafLoss)
 