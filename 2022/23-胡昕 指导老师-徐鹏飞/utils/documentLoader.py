#!/usr/bin/env python
# encoding: utf-8
# author: Hu Xin
# email: qzlyhx@gmail.com
# time: 2022.8.11
# description:
#   This file contains code that loads documens from disk through given path and file names.
#   also, the functions that shows target image with labels are included


from logging import root
from xml.dom.minicompat import NodeList
import xml.etree.ElementTree as ElementTree
import cv2
import torch
import numpy as np

from models.tree import Node, Tree
from utils.positionMapper import PositionMapper, addPositionStatistic

from utils.typeMapper import TypeMapper

class Region():
    def __init__(self, type, id=None, textColor="black", backgroundColor="white", orientation="0.00000") -> None:
        self.type = type
        self.id = id
        self.textColor = textColor
        self.backgroundColor = backgroundColor
        self.orientation = orientation
        # temperarily set coordinates as 0 length list
        # coordinates in this list are in form of (x, y) tuple
        self.coordinates = list()
        self.isNoiseRegion = False

        # including w, h of a region
        # and onehot encoding of region type
        # They are concatenated togeteher
        self.atomicUnitFeature = None
        # bbox in (minX, minY, maxX, maxY) shape
        self.bbox = None

    @staticmethod
    def fromRegionElement(element): 
        self = Region("")
        attrib = element.attrib
        self.id = attrib['id']

        try:
            self.type = attrib['type']
            self.textColor = attrib['textColour']
            self.backgroundColor = attrib['bgColour']
            self.orientation = float(attrib['orientation'])
        except:
            tag = element.tag
            tags = tag.split('}')
            tag = tags[-1]

            # to decide whether this region is a noise Region
            if tag == "NoiseRegion":
                self.isNoiseRegion = True
                self.type = "noiseRegion"
            elif tag == "ImageRegion":
                
                self.type = "imageRegion"
            elif tag == "GraphicRegion":
                self.type = "graphicRegion"

        # set coordinates
        cords = element[0]
        for point in cords:
            loc = point.attrib
            self.coordinates.append( ( int(loc['x']), int(loc['y']) ) )
        return self

    # generate features of AtomicUnit
    # return ([w, h], bounding box)
    def generateAtomicUnitFeature(self, imgWidth, imgHeight, mode = 0):
        # if having generated, return directed
        if self.atomicUnitFeature != None:
            return self.atomicUnitFeature

        # find max/min x and max/min y
        maxX, maxY = 0, 0
        minX, minY = 0x0FFFFFFF, 0x0FFFFFFF
        for cords in self.coordinates:
            if cords[0] > maxX:
                maxX = cords[0]
            if cords[0] < minX:
                minX = cords[0]
            if (cords[1] > maxY):
                maxY = cords[1]
            if (cords[1] < minY):
                minY = cords[1]
        featureLoc = None
        
        #calculate bbox
        bbox = None
        if mode == 0:
            bbox = torch.tensor([minX/imgWidth, minY/imgHeight, maxX/imgWidth, maxY/imgHeight], dtype=torch.float32)
        else:
            bbox = torch.tensor([minX/imgWidth, minY/imgHeight, minX/imgWidth, maxY/imgHeight,\
                                        maxX/imgWidth, minY/imgHeight, maxX/imgWidth, maxY/imgHeight], dtype=torch.float32)

        w = (maxX - minX)/imgWidth
        h = (maxY - minY)/imgHeight
        # only the tensor enter a tree
        featureLoc = torch.tensor([w, h], dtype=torch.float32, requires_grad=False)

        onehot, _ = TypeMapper.typeToOnehot(self.type)
        self.atomicUnitFeature =  torch.cat([featureLoc, onehot], dim=0)
        self.bbox = bbox
        # print("bbb",  self.atomicUnitFeature)
        return self.atomicUnitFeature, bbox


class Document():
    def __init__(self, imagePath=None, labelPath=None) -> None:
        self.documentImagePath = None
        self.documentImageData = None
        self.documentLabelPath = None
        self.documentLabelData = None
        self.documentTreeRootNode = None
        self.imageAttributes = None
        self.regions = None
        self.layoutTree = None

        if imagePath != None or labelPath != None:
            self.load(imagePath, labelPath)

    def load(self, imagePath, labelPath):
        if imagePath != None:
            # Only set Image Path, not load the whole image directly
            # hence image cost too mush memory
            # use getImage() function to get image data seperately
            self.documentImagePath = imagePath
        if labelPath != None:
            # load XML label file to memory
            xmlDataTree = ElementTree.parse(labelPath)
            root = xmlDataTree.getroot()
            self.documentLabelData = root
            self.documentLabelPath = labelPath
            
    def getImage(self):
        if self.documentImageData != None:
            return self.documentImageData
        img = cv2.imread(self.documentImagePath, cv2.IMREAD_COLOR)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img
    
    # Color Picker for differnt regions
    @staticmethod
    def pickColor(regionType):
        thickness = 3
        if regionType == "paragraph":
            # Red
            return (0, 0, 255), thickness
        elif regionType == "heading":
            return (255, 0, 127), thickness
        elif regionType == "floating":
            return (170, 170, 255), thickness
        elif regionType == "footer":
            return (0, 255, 0), thickness
        elif regionType == "page-number":
            return (0, 255, 255), thickness
        elif regionType == "imageRegion":
            return (232, 185, 124), thickness
        elif regionType == "graphicRegion":
            return (0, 139, 255), thickness
        elif regionType == "caption":
            return (167, 54, 233), thickness
        elif regionType == "credit":
            return (88, 201, 22), thickness
        else:
            return (255, 0, 0), thickness

    # this function returns targetmiage with label boxes in the whole image
    def getLabeledImage(self, show=True):
        img = None
        if self.documentImageData != None:
            img = self.documentImageData
        else:
            img = cv2.imread(self.documentImagePath, cv2.IMREAD_COLOR)
        # no label data
        if self.documentLabelData == None:
            return img

        # read bounding boxes from label data
        imageAttributes, regions = self.getDetailedLabels()
        
        # Draw regions on img
        color = (255 ,0 , 0)
        thickness = 9
        for region in regions:
            # do not pross noise region
            if region.type == "noiseRegion":
                continue
            # acquire type color
            color, thickness = self.pickColor(region.type)
            cods = region.coordinates
            for i in range(0, len(cods)):
                img = cv2.line(img, cods[i - 1], cods[i], color, thickness)
        
        cv2.imwrite('output/' + imageAttributes['imageFilename'], img)
        if show:
            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # get detailed image attributes and regions
    def getDetailedLabels(self):
        # Labels have already been loaded
        if self.regions != None:
            return self.imageAttributes, self.regions
        # read bounding boxes from label data
        labelData = self.documentLabelData
        imageAttributes = labelData[1].attrib
        # in form of {'imageFilename': '00000086.tif', 'imageWidth': 'xxxx', 'imageHeight': 'xxxx'}
        boundBoxes = labelData[1]
        regions = list()
        for regEle in boundBoxes:
            region = Region.fromRegionElement(regEle)
            regions.append(region)
        
        self.imageAttributes = imageAttributes
        self.regions = regions
        return imageAttributes, regions

    # get sort key of region
    @staticmethod
    def __regionSortByY(region):
        return region.bbox[1] # minY
    
    @staticmethod
    def __regionSortByX(region):
        return region.bbox[0] # minX

    #!!! Nor fully tested yet
    # This fucntion convert a Document to layoutTree that is used in training
    def generateLayoutTree(self) -> Tree:
        #if layout tree has already been generated
        if self.layoutTree != None:
            return self.layoutTree
        # detailed labels not exist, generate it
        imageAttributes, regions = None, None
        if self.regions == None: 
            imageAttributes, regions = self.getDetailedLabels()
        documentHeight = int(imageAttributes['imageHeight'])
        documentWidth = int(imageAttributes['imageWidth'])

        if documentHeight < 10:
            img = None
            if self.documentImageData != None:
                img = self.documentImageData
            else:
                img = cv2.imread(self.documentImagePath, cv2.IMREAD_COLOR)
            documentHeight = img.shape[0]
            documentWidth = img.shape[1]
        
        #calculate features
        for region in regions:
            # print(self.documentLabelPath)
            region.generateAtomicUnitFeature(documentWidth, documentHeight)

        # X as second key and Y as first key
        # to confirm that the tree was generated from up to down, left to right
        regions.sort(key=self.__regionSortByX)
        regions.sort(key=self.__regionSortByY)

        # The region that is used in RvNN
        targetRegions = list()
        for region in regions:
            _, isTargteType = TypeMapper.typeToOnehot(region.type)
            if isTargteType and len(region.coordinates) != 0:
                targetRegions.append(region)
        
        # start to generate Tree
        region0 = targetRegions[0]
        rootNow = Node(leafAtomicUnitData=region0.atomicUnitFeature.unsqueeze(dim=0), bbox=region0.bbox)
        #also first 4 numbers in rootNow.leafAtomicUnitData are bbox of root
        bboxNow = region0.bbox

        for i in range(1, len(targetRegions)):
            region = targetRegions[i]
            #generate leaf node for region
            regionNode = Node(leafAtomicUnitData=region.atomicUnitFeature.unsqueeze(dim=0), bbox=region.bbox)
            # judege relative position
            relation = PositionMapper.calculateRelativePosition(region.bbox, bboxNow)
            # get detailed relation
            detailedRelation = PositionMapper.RcalculateDetailedRelativePositionR(region.bbox, bboxNow)
            relation, _ = PositionMapper.positionToOnehot(relation)
            bboxNow = PositionMapper.merge2BBox(region.bbox, bboxNow)
            newRoot = Node(child1=rootNow, child2=regionNode, nodeRelativePositionData=relation.unsqueeze(dim=0) \
                        , nodeDetailedRelativePositionData=detailedRelation.unsqueeze(dim=0))
            rootNow = newRoot

            # print("aaa", regionNode.leafAtomicUnitData)
        
        self.layoutTree = Tree(rootNode=rootNow)
        return self.layoutTree


    # this function shows the document layout organized by a layout tree
    @staticmethod
    def showLayoutTree(tree: Tree, width=1080, height=1920, savePath = "test.png", show=True, boxoffset = 0):
        rootNode =tree.rootNode
        # blank img
        img = np.zeros((height, width, 3), np.uint8) + 255
        # Check element in trees
        nodeList = list()
        nodeList.append(rootNode)

        while len(nodeList) != 0:
            node = nodeList.pop(0)
            if node.child1 != None:
                # in case of a parent, then add childs to the queue list
                nodeList.append(node.child2)
                nodeList.append(node.child1)
            else:
                # in case of a leaf node and bbox calculated
                if (node.bbox == None):
                    continue
                bboxData = node.bbox.squeeze()[0:4]
                nodeType= TypeMapper.onehotToType(node.leafAtomicUnitData.squeeze()[-TypeMapper.getTypeNumbers():])
                # Draw bbox on img
                color, thickness = Document.pickColor(nodeType)
                
                minx, miny, maxx, maxy = bboxData[0] * width, bboxData[1] * height, bboxData[2] * width, bboxData[3] * height
                minx, miny, maxx, maxy = int(minx.item()), int(miny.item()), int(maxx.item()), int(maxy.item())
                minx, miny, maxx, maxy = minx + boxoffset, miny+boxoffset, maxx-boxoffset, maxy-boxoffset
                cords = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
                for i in range(0, len(cords)):
                    img = cv2.line(img, cords[i - 1], cords[i], color, thickness)
        cv2.imwrite(savePath, img)
        if show:
            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # this function regenerates the BBox of a tree that is generated by decoder
    @staticmethod
    def restoreBBox(targetTree:Tree):
        # use the last leaf's(first bbox that fed into Tree) minX, minY as (0, 0), calculate offset
        rootNode = targetTree.rootNode
        if rootNode.child2 == None:
            return

        nodeList = []
        while rootNode.child1 != None:
            nodeList.append(rootNode)
            nodeList.append(rootNode.child2)
            rootNode = rootNode.child1
        #last Leaf
        lleaf = rootNode
        if lleaf.leafAtomicUnitData == None:
            print("err Tree, unable to restore")
            return
        # w and h
        bboxNow = torch.tensor([0, 0, lleaf.leafAtomicUnitData[0][0], lleaf.leafAtomicUnitData[0][1]], dtype=torch.float32)
        lleaf.bbox = bboxNow
        # regenerate bbox from Leafs to root
        for i in range(0, len(nodeList)//2):
            index = len(nodeList) - i * 2 - 1
            indexRela = index - 1
            leaf = nodeList[index]
            node = nodeList[indexRela]
            leaf.bbox = PositionMapper.RcalculateBBoxThroughDetailedPositionR(\
                node.nodeDetailedRelativePositionData.squeeze(), bboxNow, leaf.leafAtomicUnitData[0][0],leaf.leafAtomicUnitData[0][1]\
                )
            bboxNow = PositionMapper.merge2BBox(bboxNow,leaf.bbox)
            node.bbox = bboxNow
        
        # Then all node had been merged
        # calculate offset
        offset = [bboxNow[0], bboxNow[1], bboxNow[2] - bboxNow[0], bboxNow[3] - bboxNow[1]]
        # use [0, 0] offset temprarily
        # offset = [0.02, 0.02]
        PositionMapper.applyOffset(lleaf.bbox, offset)
        # print(bboxNow, "  ", offset)
        leafList = []
        leafList.append(lleaf)
        for i in range(0, len(nodeList)//2):
            index = len(nodeList) - i * 2 - 1
            leaf = nodeList[index]
            leafList.append(leaf)
            PositionMapper.applyOffset(leaf.bbox, offset)
        # print(nodeList[0])

        # Now all BBox in tree has been calculated
        bboxList = []
        for i in range(0, len(leafList)):
            newbbox = 0
            leafnode = leafList[i]
            lbbox = leafnode.bbox
            type = TypeMapper.onehotToType(leafnode.leafAtomicUnitData.squeeze()[-TypeMapper.getTypeNumbers():])
            type = TypeMapper.getTypeId(type)
            x1, y1, x2, y2 = lbbox[0].item(), lbbox[1].item(), lbbox[2].item(), lbbox[3].item()
            newbbox = [type] + [x1, y1, x2-x1, y2-y1]
            if newbbox[3] > 0 and newbbox[4] > 0:
                bboxList.append(newbbox)
        return bboxList

