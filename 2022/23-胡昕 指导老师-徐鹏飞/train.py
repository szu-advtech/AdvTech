#!/usr/bin/env python
# encoding: utf-8
# author: Hu Xin
# email: qzlyhx@gmail.com
# time: 2022.8.15
# description:
#   This file is runnable - run this file to train

import argparse

from models.loss import calculateLossBetweenTrees
from models.recursiveDecoder import RecursiveDecoder
from models.tree import Node, Tree, TreeStructureAnalyser
import utils.config as Configuration
import models.recursiveEncoder as Encoder
import torch
from torch.utils.data.dataloader import DataLoader
import models.sampler as Sampler
from utils.datasetLoader import DocumentTestDataset, DocumentTrainDataset, LayoutTestDataset, LayoutTrainDataset, collate_fn
from utils.modelManager import ModelManager
from utils.documentLoader import Document
import os
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    Configuration.addArgs(parser)
    config = parser.parse_args()
    config = Configuration.configProcess(config)
    # copy learning parameters from confin
    batchSize = config.batchSize
    lr = config.learningRate
    epochNum = config.epoch
    datasetRoot = config.datasetRoot

    # load Models
    encoder = Encoder.RecursiveEncoder(config)
    decoder = RecursiveDecoder(config)
    sampler = Sampler.Sampler(config.encoderOutputSize, config.decoderInputSize)

    ModelManager.encoderInit(encoder, config, loadCheckPoint=False)
    ModelManager.samplerInit(sampler, config, loadCheckPoint=False)
    ModelManager.decoderInit(decoder, config, loadCheckPoint=False)

    # load dataset
    # trainDataset = DocumentTrainDataset(root_dir=datasetRoot)
    trainDataset = LayoutTrainDataset(root_dir="./publaynet/")
    datasetLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
    # test dataset
    # testDataset = DocumentTestDataset(root_dir=datasetRoot)
    testDataset = LayoutTestDataset(root_dir="./publaynet/")
    testDatasetLoader = DataLoader(testDataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    # get Parameters and load optimizer
    encoderParas = encoder.getPatameters()
    samplerParas = sampler.getParameters()
    decoderParas = decoder.getParameters()
    optimizer = torch.optim.Adam(encoderParas + samplerParas + decoderParas, lr=lr)
    lrScheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    # for p in decoderParas:
    #     for i in p['params']:
    #         print(i)
    # print("---")
    
    # start to Train
    print("[status]Now start to train Network - ")
    # for log
    writer = SummaryWriter("./run/expr_1")
    epochOffset = 0
    globalStep = 0
    #epoch
    for e in range(0, epochNum):
        print("[status] Reaching a new Epoch-{}".format(e))
        # 
        for i, data in enumerate(datasetLoader):
            # calculate docs in batches individually
            # Four list of losses
            kldLList, leafRecLList, relativePosLList, ceLList, isleafLList = [], [], [], [], []
            loss = 0

            testGTTree = None
            testOutTree = None

            globalStep = globalStep + 1
            docCount = 0
            for doc in data:
                docCount = docCount + 1
                # generate tree
                gtTree = doc.generateLayoutTree()
                # fed into network
                encodedFeatures =  encoder.encodeATree(gtTree)
                featureInLatentSpace, kld = sampler.forward(encodedFeatures)
                resultTree = decoder.decodeToTree(featureInLatentSpace, gtTree=gtTree)
                # calculate Loss
                leafCLoss, relativePosLoss, ceLoss, isLeafLoss=calculateLossBetweenTrees(resultTree, gtTree)
                # append Losses
                kldLList.append(kld)
                leafRecLList.append(leafCLoss)
                relativePosLList.append(relativePosLoss)
                ceLList.append(ceLoss)
                isleafLList.append(isLeafLoss)
                testGTTree = gtTree
                testOutTree = featureInLatentSpace
                # print("old-", TreeStructureAnalyser.calculateDepthWidth(gtTree.rootNode))
                # print("New-", TreeStructureAnalyser.calculateDepthWidth(resultTree.rootNode))
                # print(kld, leafCLoss, relativePosLoss, ceLoss)

            
            # Test result
            testOutTree = decoder.decodeToTree(featureInLatentSpace, gtTree=None)
            Document.restoreBBox(testOutTree)
            Document.showLayoutTree(testGTTree, show=False, savePath="./validation/test-e{}-s{}-gt.png".format(str(e), str(i)) )
            Document.showLayoutTree(testOutTree, show=False, savePath="./validation/test-e{}-s{}-re.png".format(str(e), str(i)) )
            # Then aggregate all losses for gradient
            kld = sum(kldLList) / docCount
            leafCLoss = sum(leafRecLList) / docCount
            relativePosLoss = sum(relativePosLList) / docCount
            ceLoss = sum(ceLList) / docCount
            isLeafLoss = sum(isleafLList) / docCount

            lossAddup =  kld + 5 * leafCLoss + 7.5 * relativePosLoss + 10 * ceLoss + 15 * isLeafLoss

            print("[status]step {} Losses:\n kld:{}\n Leaf Contruction Loss:{}\n Relative Position Loss:{}\n CE Loss:{}\n isLeaf Loss:{}\n"\
                  .format(i, kld, leafCLoss, relativePosLoss, ceLoss, isLeafLoss) )
            
            writer.add_scalar("kld", kld, globalStep)
            writer.add_scalar("leafCLoss", leafCLoss, globalStep)
            writer.add_scalar("relativePosLoss", relativePosLoss, globalStep)
            writer.add_scalar("ceLoss", ceLoss, globalStep)
            writer.add_scalar("isLeafLoss", isLeafLoss, globalStep)

            loss = lossAddup.item()
            # backward and train
            optimizer.zero_grad()
            lossAddup.backward() 

            optimizer.step()
        
        #schedular step
        if (e < 35):
            lrScheduler.step()

        # Validating
        # if e % 3 == 0:
            # kldLList, leafRecLList, relativePosLList, ceLList, isleafLList = [], [], [], [], []
            # loss = 0
            # sumnum = 0
            # for i, data in enumerate(datasetLoader):
            #     for doc in data:
            #         sumnum = sumnum + 1
            #         gtTree = doc.generateLayoutTree()
            #         encodedFeatures =  encoder.encodeATree(gtTree)
            #         featureInLatentSpace, kld = sampler.forward(encodedFeatures)
            #         resultTree = decoder.decodeToTree(featureInLatentSpace, gtTree=gtTree)
            #         leafCLoss, relativePosLoss, ceLoss, isLeafLoss=calculateLossBetweenTrees(resultTree, gtTree)
            #         kldLList.append(kld)
            #         leafRecLList.append(leafCLoss)
            #         relativePosLList.append(relativePosLoss)
            #         ceLList.append(ceLoss)
            #         isleafLList.append(isLeafLoss)
            # kld = sum(kldLList) / sumnum
            # leafCLoss = sum(leafRecLList) / sumnum
            # relativePosLoss = sum(relativePosLList) / sumnum
            # ceLoss = sum(ceLList) / sumnum
            # isLeafLoss = sum(isleafLList) / sumnum
            # # add Losses to writer
            # writer.add_scalar("kld", kld, e + epochOffset)
            # writer.add_scalar("leafCLoss", leafCLoss, e + epochOffset)
            # writer.add_scalar("relativePosLoss", relativePosLoss, e + epochOffset)
            # writer.add_scalar("ceLoss", ceLoss, e + epochOffset)
            # writer.add_scalar("isLeafLoss", isLeafLoss, e + epochOffset)

            #test dataset
            # ldLList, leafRecLList, relativePosLList, ceLList, isleafLList = [], [], [], [], []
            # loss = 0
            # sumnum = 0
            # for i, data in enumerate(testDatasetLoader):
            #     for doc in data:
            #         sumnum = sumnum + 1
            #         gtTree = doc.generateLayoutTree()
            #         encodedFeatures =  encoder.encodeATree(gtTree)
            #         featureInLatentSpace, kld = sampler.forward(encodedFeatures)
            #         resultTree = decoder.decodeToTree(featureInLatentSpace, gtTree=gtTree)
            #         leafCLoss, relativePosLoss, ceLoss, isLeafLoss=calculateLossBetweenTrees(resultTree, gtTree)
            #         kldLList.append(kld)
            #         leafRecLList.append(leafCLoss)
            #         relativePosLList.append(relativePosLoss)
            #         ceLList.append(ceLoss)
            #         isleafLList.append(isLeafLoss)
            # kld = sum(kldLList) / sumnum
            # leafCLoss = sum(leafRecLList) / sumnum
            # relativePosLoss = sum(relativePosLList) / sumnum
            # ceLoss = sum(ceLList) / sumnum
            # isLeafLoss = sum(isleafLList) / sumnum
            # writer.add_scalar("test-kld", kld, globalStep)
            # writer.add_scalar("test-leafCLoss", leafCLoss, globalStep)
            # writer.add_scalar("test-relativePosLoss", relativePosLoss, globalStep)
            # writer.add_scalar("test-ceLoss", ceLoss, globalStep)
            # writer.add_scalar("test-isLeafLoss", isLeafLoss, globalStep)
                    

        # Save parameters
        if e % 3 == 0:
            ModelManager.encoderSave(encoder, config, "epoch-" + str(e) + " " + str(loss))
            ModelManager.samplerSave(sampler, config, "epoch-" + str(e) + " " + str(loss))
            ModelManager.decoderSave(decoder, config, "epoch-" + str(e) + " " + str(loss))
    writer.close()




