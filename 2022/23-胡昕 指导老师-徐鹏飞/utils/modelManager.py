#!/usr/bin/env python
# encoding: utf-8
# author: Hu Xin
# email: qzlyhx@gmail.com
# time: 2022.8.11
# description:
#   This file implemnts the Model Initializer that does
#   1. initialize parameters in model if not load from check point
#   2. load models from check point if the check point is specified
#   and the Model Saver that
#   1. save models to target disk location


import torch
import os

from models.sampler import Sampler
from utils.config import getDevice

class ModelManager():
    # This function sava all parameters in encoder into 1 file - "encoder.pt"
    @staticmethod
    def encoderInit(encoder, config, loadCheckPoint=False):
        if not loadCheckPoint:
            # initialize the weights sampled from a Gaussian distribution
            encoder.leafUnit.initParameters(torch.nn.init.normal_)
            for key in encoder.encoderUnit:
                encoder.encoderUnit[key].initParameters(torch.nn.init.normal_)
        else:
            # load check point
            loadPath = config.checkPointPath
            modelName = "encoder.pt"
            device = getDevice()
            checkPoint = torch.load(loadPath + modelName, map_location=device)
            encoder.leafUnit.load_state_dict(checkPoint['leaf_unit_dict'])
            for key in encoder.encoderUnit:
                encoder.encoderUnit[key].load_state_dict(checkPoint[key + "_encoder_unit_dict"])
                encoder.encoderUnit[key].eval()
            encoder.leafUnit.eval()
            
    
    @staticmethod
    def encoderSave(encoder, config, extraPath="run"):
        #save model paremeters to disk
        saveName = "encoder.pt"
        savePath = config.experimentSavePath + extraPath + "/"
        os.makedirs(savePath, exist_ok=True)
        saveDict = dict()
        saveDict["leaf_unit_dict"] = encoder.leafUnit.state_dict()
        for key in encoder.encoderUnit:
            saveDict[key + "_encoder_unit_dict"] = encoder.encoderUnit[key].state_dict()

        torch.save(
            saveDict,
            savePath + saveName
        )

    @staticmethod
    def samplerInit(sampler, config, loadCheckPoint=False):
        if not loadCheckPoint:
            # initialize the weights sampled from a Gaussian distribution
            sampler.initParameters(torch.nn.init.normal_)
        else:
            # load check point
            loadPath = config.checkPointPath
            modelName = "sampler.pt"
            device = getDevice()
            checkPoint = torch.load(loadPath + modelName, map_location=device)
            sampler.load_state_dict(checkPoint['sampler_dict'])
            sampler.eval()
    
    @staticmethod
    def samplerSave(sampler, config, extraPath="run"):
        #save model paremeters to disk
        saveName = "sampler.pt"
        savePath = config.experimentSavePath + extraPath + "/"
        os.makedirs(savePath, exist_ok=True)
        torch.save(
            {
                'sampler_dict': sampler.state_dict(),
            },
            savePath + saveName
        )
    
    @staticmethod
    def decoderInit(decoder, config, loadCheckPoint=False):
        if not loadCheckPoint:
            # initialize the weights sampled from a Gaussian distribution
            for key in decoder.decoderUnit:
                decoder.decoderUnit[key].initParameters(torch.nn.init.normal_)
            decoder.featureMapBackUnit.initParameters(torch.nn.init.normal_)
            decoder.classifierUnit.initParameters(torch.nn.init.normal_)
            decoder.leafClassifierUnit.initParameters(torch.nn.init.normal_)
        else:
            # load check point
            loadPath = config.checkPointPath
            modelName = "decoder.pt"
            device = getDevice()
            checkPoint = torch.load(loadPath + modelName, map_location= device)
            for key in decoder.decoderUnit:
                decoder.decoderUnit[key].load_state_dict(checkPoint[key + '_decoder_unit_dict'])
                decoder.decoderUnit[key].eval()
            decoder.featureMapBackUnit.load_state_dict(checkPoint['feature_map_back_unit_dict'])
            decoder.classifierUnit.load_state_dict(checkPoint['classifier_unit_dict'])
            decoder.leafClassifierUnit.load_state_dict(checkPoint['leaf_classifier_unit_dict'])
            
            decoder.featureMapBackUnit.eval()
            decoder.classifierUnit.eval()
    
    @staticmethod
    def decoderSave(decoder, config, extraPath="run"):
        #save model paremeters to disk
        saveName = "decoder.pt"
        savePath = config.experimentSavePath + extraPath + "/"
        os.makedirs(savePath, exist_ok=True)

        saveDict = dict()
        saveDict["feature_map_back_unit_dict"] = decoder.featureMapBackUnit.state_dict()
        saveDict["classifier_unit_dict"] = decoder.classifierUnit.state_dict()
        saveDict["leaf_classifier_unit_dict"] = decoder.leafClassifierUnit.state_dict()
        for key in decoder.decoderUnit:
            saveDict[key + "_decoder_unit_dict"] = decoder.decoderUnit[key].state_dict()

        torch.save(
            saveDict,
            savePath + saveName
        )
        
            


