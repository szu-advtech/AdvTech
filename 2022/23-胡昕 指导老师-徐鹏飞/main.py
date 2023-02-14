import argparse
from models.loss import calculateLossBetweenTrees
from models.recursiveDecoder import RecursiveDecoder
from models.tree import Node, Tree, TreeStructureAnalyser
import utils.config as Configuration
import models.recursiveEncoder as Encoder
import torch
import models.sampler as Sampler
from utils.modelManager import ModelManager
from utils.documentLoader import Document

#test func
def generateATreeData(config) -> Tree:
    rootNode = Node(nodeRelativePositionData=torch.randn([1,config.relativePositionSize], dtype=torch.float32))
    rootNode.setChild1(Node(leafAtomicUnitData=torch.randn([1,config.atomicUnitSize], dtype=torch.float32)))
    rootNode.setChild2(Node(leafAtomicUnitData=torch.randn([1,config.atomicUnitSize], dtype=torch.float32)))
    tTree = Tree(rootNode=rootNode)
    return tTree


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    Configuration.addArgs(parser)
    config = parser.parse_args()
    config = Configuration.configProcess(config)
    encoder = Encoder.RecursiveEncoder(config)
    doc = Document(imagePath="D:/datasets/PRImA Layout Analysis Dataset/Images/00000272.tif", labelPath="D:/datasets/PRImA Layout Analysis Dataset/XML/00000272.xml")
    testTree = doc.generateLayoutTree()
    # Document.showLayoutTree(testTree, savePath="test.png")

    #init encoder & decoder
    ModelManager.encoderInit(encoder, config, False)
    
    # print("Model's state_dict:")
    # for param_tensor in encoder.state_dict():
    #     print(param_tensor, "\t", encoder.state_dict()[param_tensor].size())
    # x = torch.randn([10,614], dtype=torch.float32)
    # print(encoder.forward(x).size())

    encodedFeatures =  encoder.encodeATree(testTree)
    sampler = Sampler.Sampler(config.encoderOutputSize, config.decoderInputSize)

    print(encoder.leafUnit.parameters())

    ModelManager.samplerInit(sampler, config, False)

    featureInLatentSpace, kld = sampler.forward(encodedFeatures)
    #reconstruct tree
    rd = RecursiveDecoder(config)
    ModelManager.decoderInit(rd, config, False)
    newTree = rd.decodeToTree(featureInLatentSpace)
    # Document.showLayoutTree(newTree, savePath="testOutput.png")
    # print(newTree.rootNode.nodeData)
    print(TreeStructureAnalyser.calculateDepthWidth(newTree.rootNode))

    # to see loss on a single tree
    kldLoss = kld
    leafCLoss, relativePosLoss, ceLoss=calculateLossBetweenTrees(newTree, testTree)
    print("Losses:\n kld:{}\n Leaf Contruction Loss:{}\n Relative Position Loss:{}\n CE Loss:{}\n".format(kld, leafCLoss, relativePosLoss, ceLoss) )

    #save models+
    # ModelManager.encoderSave(encoder, config, "testRun")
    # ModelManager.samplerSave(sampler, config, "testRun")
    # ModelManager.decoderSave(rd, config, "testRun")

