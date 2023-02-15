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
from torch.utils.tensorboard import SummaryWriter


def saveBoxes(boxes, path):
    box_str = ""
    for i in range(len(boxes)):
        box = boxes[i]
        newstr = "{} {} {} {} {}\n".format(int(box[0]), box[1], box[2], box[3], box[4])
        box_str += newstr
    with open(path, "w") as f:
        f.write(box_str)
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    Configuration.addArgs(parser)
    config = parser.parse_args()
    config = Configuration.configProcess(config)


    encoder = Encoder.RecursiveEncoder(config)
    decoder = RecursiveDecoder(config)
    sampler = Sampler.Sampler(config.encoderOutputSize, config.decoderInputSize)

    ModelManager.encoderInit(encoder, config, loadCheckPoint=True)
    ModelManager.samplerInit(sampler, config, loadCheckPoint=True)
    ModelManager.decoderInit(decoder, config, loadCheckPoint=True)
    # set cpu
    config.device = "cuda:0"
    datasetRoot = config.datasetRoot
    trainDataset = LayoutTestDataset(root_dir="D:/datasets/GTLayoutNew/publaynet/publayNew/publay_new_1110/")
    datasetLoader = DataLoader(trainDataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    count = 0
    countmax = 2000

    saveRoot = "./samples/"

    with torch.no_grad():
        for i, data in enumerate(datasetLoader):
            if count > countmax:
                break
            for doc in data:
                gtTree = doc.generateLayoutTree()
                # fed into network
                encodedFeatures =  encoder.encodeATree(gtTree)
                featureInLatentSpace, kld = sampler.forward(encodedFeatures)
                resultTree = decoder.decodeToTree(featureInLatentSpace)
                docname = doc.documentLabelPath
                # save results
                bboxList1 = Document.restoreBBox(gtTree)
                bboxList2 = Document.restoreBBox(resultTree)
                Document.showLayoutTree(gtTree, show=False, savePath="{}READ_publaynet_train_supli/{}_GT.png".format(saveRoot, docname) )
                Document.showLayoutTree(resultTree, show=False, savePath="{}READ_publaynet_train_supli/{}_PRED.png".format(saveRoot, docname))
                saveBoxes(bboxList1, "{}READ_publaynet_train_supli/{}_GT.txt".format(saveRoot, docname))
                saveBoxes(bboxList2, "{}READ_publaynet_train_supli/{}_PRED.txt".format(saveRoot, docname))
                if count > countmax:
                    break
                count+=1


    # doc0 = None
    # for i, data in enumerate(datasetLoader): 
    #     for doc in data:
    #         if doc0 != None:
    #             gtTree0 = doc0.generateLayoutTree()
    #             gtTree = doc.generateLayoutTree()
    #             # fed into network
    #             encodedFeatures0 =  encoder.encodeATree(gtTree)
    #             encodedFeatures =  encoder.encodeATree(gtTree)

    #             for t in range(0, 11):
    #                 featureInLatentSpace = sampler.mix2(encodedFeatures0, encodedFeatures, t/10)
    #                 resultTree = decoder.decodeToTree(featureInLatentSpace, gtTree=gtTree)
    #                 # save results
    #                 Document.restoreBBox(resultTree)
    #                 Document.showLayoutTree(resultTree, show=False, savePath="./output-test/mix-{}-re{}.png".format(str(i), str(t/10)) )
    #             Document.showLayoutTree(gtTree0, show=False, savePath="./output-test/mix-{}-gt0.png".format(str(i)) )
    #             Document.showLayoutTree(gtTree, show=False, savePath="./output-test/mix-{}-gt1.png".format(str(i)) )
                
    #         doc0 = doc

    # for i, data in enumerate(datasetLoader): 
    #     for doc in data:
    #         doc0 = doc

    # for i in range(0, 1001):
    #         sampleFeatLength = 300
    #         feat = torch.randn(1, sampleFeatLength)
    #         resultTree = decoder.decodeToTree(feat)
    #         # save results
    #         bboxes = Document.restoreBBox(resultTree)
    #         #Document.showLayoutTree(gtTree, show=False, savePath="./output-test/{}-gt.png".format(str(i)) )
    #         if bboxes:
    #             Document.showLayoutTree(resultTree, show=False, savePath="{}READ_publaynet_generate/{}.png".format(saveRoot, str(i)) )
    #             saveBoxes(bboxes, "{}READ_publaynet_generate/{}.txt".format(saveRoot, str(i)))

