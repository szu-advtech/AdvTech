#!/usr/bin/env python
# encoding: utf-8
# author: Hu Xin
# email: qzlyhx@gmail.com
# time: 2022.8.12
# description:
#   This file contains regions types's name and the funtions that convert variables between str region type and type onehot encoding

import torch

# These types should match withn those loaded from document Loader
regionTypes = [
    'heading',
    'paragraph',
    'imageRegion',
    'list',
    'table'
]

regionTypeNum = len(regionTypes)

class TypeMapper():
    # returns (onehot encoding, strInTypes)
    @staticmethod
    def typeToOnehot(typeName):
        onehot = torch.zeros([regionTypeNum], dtype=torch.float32)
        for i in range(0, len(regionTypes)):
            if regionTypes[i] == typeName:
                #matched
                onehot[i] = 1
                return onehot, True
        #not find
        return onehot, False
    
    @staticmethod
    def getTypeId(typStr):
        for i in range(len(regionTypes)):
            if typStr == regionTypes[i]:
                return i
        return 0
        
    @staticmethod
    def onehotToType(onehot):
        # for i in range(0, onehot.size()[-1]):
        #     if torch.abs(onehot[i] - 1) < 0.001:
        #         return  regionTypes[i]
        # return ""
        # select 1 max value
        max, index = -0x0FFFFFFF, 0
        for i in range(0, onehot.size()[-1]):
            if onehot[i] > max:
                max, index = onehot[i], i
        return regionTypes[index]
    
    def getTypeNumbers():
        return regionTypeNum