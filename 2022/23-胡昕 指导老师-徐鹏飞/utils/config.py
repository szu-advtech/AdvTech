#!/usr/bin/env python
# encoding: utf-8
# author: Hu Xin
# email: qzlyhx@gmail.com
# time: 2022.8.9
# description:
#   This file contains basic configurations of the whole project, in form of argument parser

from argparse import ArgumentParser
from unittest import defaultTestLoader
from datetime import datetime
from utils.positionMapper import PositionMapper
from utils.typeMapper import TypeMapper

# Global variables
# Due to the design of the whole network structure, put device here
# use getDevice() to access
device = "cpu"


# this function add args to argparser & set default values of args used in the program
def addArgs(parser:ArgumentParser)-> ArgumentParser: 

    parser.add_argument('--device', type=str, default= "cuda:0", help='')
    # Training parameters
    parser.add_argument('--epoch', type=int, default= 100000, help='')
    parser.add_argument('--batchSize', type=int, default= 128, help='')
    parser.add_argument('--learningRate', type=float, default=1e-3, help='')
    parser.add_argument('--datasetRoot', type=str, default= "D:/datasets/PRImA Layout Analysis Dataset/", help='')


    # dataset root path
    parser.add_argument('--datasetRootPath', type=str, default= "D:/datasets/PRImA Layout Analysis Dataset/", help='')
    # parameters for model saving&loading path
    parser.add_argument('--checkPointPath', type=str, default= "./", help='')
    parser.add_argument('--savePath', type=str, default= "./output/", help='')

    # get type length and position length
    typeNum = TypeMapper.getTypeNumbers()
    positionNum = PositionMapper.getTypeNumbers()
    # Parameters for the network
    parser.add_argument('--encoderHiddenLayerSize', type=int, default=450, help='') # !! 不太确定原论文中间层特征长度
    parser.add_argument('--decoderHiddenLayerSize', type=int, default=450, help='') # !! 不太确定原论文中间层特征长度
    parser.add_argument('--nodeFeatureSize', type=int, default= 300, help='')
    parser.add_argument('--relativePositionSize', type=int, default= positionNum, help='')
    parser.add_argument('--detailedRelativePositionSize', type=int, default= 10, help='')

    parser.add_argument('--decoderMaxDepth', type=int, default=100, help='')
    # 待调整
    parser.add_argument('--atomicUnitSize', type=int, default= 2 + typeNum, help='')
    parser.add_argument('--maxTreeDepth', type=int, default= 27, help='')
    parser.add_argument('--classifierHiddenLayerSize', type=int, default= 100, help='')

    return  parser


# This funciton add arguments that can be calculated from args read in
def configProcess(config:object)->object:
    # experiment save path, apend date and time information after experimentPaths
    from datetime import datetime
    runDate = datetime.now()
    datestr = runDate.strftime("%y.%m.%d %H-%M-%S")

    config.experimentSavePath = config.savePath + datestr + "/"

    # extra net parameters
    config.encoderInputSize = config.nodeFeatureSize * 2 + config.detailedRelativePositionSize
    config.encoderOutputSize = config.nodeFeatureSize
    config.decoderInputSize = config.nodeFeatureSize
    config.decoderOutputSize = config.nodeFeatureSize * 2 + config.detailedRelativePositionSize
    global device 
    device = config.device
    return config

def getDevice():
    return device
