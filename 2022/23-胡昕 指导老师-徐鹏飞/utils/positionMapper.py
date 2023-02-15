#!/usr/bin/env python
# encoding: utf-8
# author: Hu Xin
# email: qzlyhx@gmail.com
# time: 2022.8.12
# description:
#   This file contains reletive potions types's name and the funtions that convert variables between str position type and onehot encoding


import torch

# Reletive positions defined here
relativePositionTypes = [
    'enclosed',
    'left',
    'bottomLeft',
    'bottom',
    'right',
    'bottomRight',
    'wideBottom'
]

relativePositionTypeNum = len(relativePositionTypes)

positionStatistic = {i:0 for i in relativePositionTypes}

def addPositionStatistic(typeName):
    positionStatistic[typeName] += 1

def getPositionStatistic():
    return positionStatistic

class PositionMapper():
    @staticmethod
    def positionToOnehot(posStr):
        onehot = torch.zeros([relativePositionTypeNum], dtype=torch.float32)
        for i in range(0, len(relativePositionTypes)):
            if relativePositionTypes[i] == posStr:
                #matched
                onehot[i] = 1
                return onehot, True
        #not find
        return onehot, False

    @staticmethod
    def onehotToPosition(onehot):
        # select 1 max value
        max, index = -0x0FFFFFFF, 0
        for i in range(0, onehot.size()[-1]):
            if onehot[i] > max:
                max, index = onehot[i], i
        return relativePositionTypes[index]
    
    @staticmethod
    def getTypeNumbers():
        return relativePositionTypeNum

    @staticmethod
    def getTypes():
        return relativePositionTypes

    # functions used in calculating relative position
    # @staticmethod
    # def __getPositionCode(x, y, referenceBox):
    #     codex, codey = 0, 0
    #     if x >= referenceBox[0] and x <= referenceBox[2]: # x > minX and x < maxX
    #         codex = 1
    #     if x > referenceBox[2]:
    #         codex = 2
    #     if y > referenceBox[3]: # y > maxY
    #         codey = 1
    #     return codex, codey

    # this function calculates bbox1's relative position on bbox2, returns str in relativePositionTypes
    @staticmethod
    def calculateRelativePosition(bbox1, bbox2) -> str:
        # get reletive position codes for 4 points in box1
        # x0, x1, x2, x3 = bbox1[0], bbox1[2], bbox1[0], bbox1[2]
        # y0, y1, y2, y3 = bbox1[1], bbox1[3], bbox1[1], bbox1[3]
        # cx0, cy0 = PositionMapper.__getPositionCode(x0, y0, bbox2) # codex, codey of x0, y0
        # cx1, cy1 = PositionMapper.__getPositionCode(x1, y1, bbox2)
        # cx2, cy2 = PositionMapper.__getPositionCode(x2, y2, bbox2)
        # cx3, cy3 = PositionMapper.__getPositionCode(x3, y3, bbox2)
        # in case of wide-bottom
        minx1, minx2, miny1, miny2 = bbox1[0], bbox2[0], bbox1[1], bbox2[1]
        maxx1, maxx2, maxy1, maxy2 = bbox1[2], bbox2[2], bbox1[3], bbox2[3]
        # margin to dealing with small pixel-level err
        margin = 0.0025 #about 5px
        minx2, miny2 = minx2 - margin, miny2 - margin
        maxx2, maxy2 = maxx2 + margin, maxy2 + margin
        w2, h2 = maxx2 - minx2, maxy2 - miny2
        interPara = 0.1
        if minx1 < minx2 and maxx1 > maxx2:
            return 'wideBottom'

        # enclosed and bottom
        if minx1 > minx2 and maxx1 < maxx2:
            if maxy1 < maxy2:
                # enclosed
                return 'enclosed'
            else:
                # in this case cy3 must be 1, the relationship is bottom
                return 'bottom'
        
        #bottomleft and bottomright
        if miny1 > maxy2:
            if minx1 < minx2:
                return "bottomLeft"
            if maxx1 > maxx2:
                return 'bottomRight'
        
        # left and right
        if miny1 < maxy2:
            if maxx1 < minx2 + w2*interPara:
                return 'left'
            if minx1 > maxx2 - w2*interPara:
                return 'right'
        
        # left enclosed condition 
        return 'enclosed'
    
    # this function calculates the DetailedRelativePosition between two bbox
    # use the relationship decribed in GRAIN
    @staticmethod
    def calculateDetailedRelativePosition(bbox1, bbox2) -> torch.Tensor:
        detailedRP = torch.zeros([20], requires_grad=False, dtype=torch.float32)
        # 4 bit distance, 4x4=16 bit relation
        # bbox2 left edge
        distance, type = PositionMapper.calculateEdgeDistance(bbox1[0], bbox2[0], bbox2[2])
        detailedRP[0], detailedRP[4 + type] = distance, 1
        distance, type = PositionMapper.calculateEdgeDistance(bbox1[2], bbox2[0], bbox2[2])
        detailedRP[1], detailedRP[8 + type] = distance, 1
        distance, type = PositionMapper.calculateEdgeDistance(bbox1[1], bbox2[1], bbox2[3])
        detailedRP[2], detailedRP[12 + type] = distance, 1
        distance, type = PositionMapper.calculateEdgeDistance(bbox1[3], bbox2[1], bbox2[3])
        detailedRP[3], detailedRP[16 + type] = distance, 1
        return detailedRP

    #4 bit relative position
    @staticmethod
    def ZcalculateDetailedRelativePositionZ(bbox1, bbox2) -> torch.Tensor:
        detailedRP = torch.zeros([4], requires_grad=False, dtype=torch.float32)
        # 4 bit distance, dx, dy / dx1, dx2
        # get TYPE
        type = PositionMapper.calculateRelativePosition(bbox1, bbox2)
        if type == "left":
            detailedRP[0], detailedRP[1] = bbox2[0] - bbox1[2], bbox2[3] - bbox1[1]
        if type == "enclosed":
            detailedRP[0], detailedRP[1] = bbox2[0] - bbox1[0], bbox2[1] - bbox1[1]
        if type == "right":
            detailedRP[0], detailedRP[1] = bbox2[2] - bbox1[0], bbox2[3] - bbox1[1]
        if type == "bottomLeft":
            detailedRP[0], detailedRP[1] = bbox2[0] - bbox1[0], bbox2[3] - bbox1[1]
        if type == "bottom":
            detailedRP[0], detailedRP[1] = bbox2[0] - bbox1[0], bbox2[3] - bbox1[1]
        if type == "bottomRight":
            detailedRP[0], detailedRP[1] = bbox2[2] - bbox1[2], bbox2[3] - bbox1[1]
        if type == "wideBottom":
            detailedRP[0], detailedRP[1] = bbox2[0] - bbox1[0], bbox2[3] - bbox1[1]
            detailedRP[2], detailedRP[3] = bbox2[0] - bbox1[0], bbox2[2] - bbox1[2]    
        return detailedRP


    # 14 bit relative position
    # first 2 bit - distance from x and y
    # last 12 bit - distant type
    @staticmethod
    def RcalculateDetailedRelativePositionR(bbox1, bbox2) -> torch.Tensor:
        detailedRP = torch.zeros([10], requires_grad=False, dtype=torch.float32)
        featx = PositionMapper.__R_calculate_feat_1dim__(bbox1[0], bbox1[2], bbox2[0], bbox2[2])
        featy = PositionMapper.__R_calculate_feat_1dim__(bbox1[1], bbox1[3], bbox2[1], bbox2[3])
        detailedRP[0] = featx[0]
        detailedRP[1] = featy[0]
        detailedRP[2:6] = featx[1:5]
        detailedRP[6:10] = featy[1:5]
        return detailedRP

    # used in RcalculateDetailedRelativePositionR, calculate feat in 1 dimension
    @staticmethod
    def __R_calculate_feat_1dim__(min1, max1, min2, max2):
        feat = torch.zeros([5], requires_grad=False, dtype=torch.float32)
        refer = [min2, max2]
        disArr = [i - min1 for i in refer] + [i - max1  for i in refer]
        minIndex, minRange = 0, abs(disArr[0])
        for i in range(1, len(disArr)):
            if abs(disArr[i]) < minRange:
                minIndex, minRange = i, abs(disArr[i])
        feat[0] = disArr[minIndex]
        feat[1 + minIndex] = 1
        return feat

    # 14 bit version of calculateBBoxThroughDetailedPosition
    @staticmethod
    def RcalculateBBoxThroughDetailedPositionR(detailedRP, bboxChild1, w, h):
        newBBox = torch.zeros([4], dtype=torch.float32)
        boxX = PositionMapper.__getbbox_1dim__(detailedRP[0], detailedRP[2:6], bboxChild1[0], bboxChild1[2], w)
        boxY = PositionMapper.__getbbox_1dim__(detailedRP[1], detailedRP[6:10], bboxChild1[1], bboxChild1[3], h)
        newBBox[0], newBBox[1], newBBox[2], newBBox[3] = boxX[0], boxY[0], boxX[1], boxY[1]
        return newBBox

    @staticmethod
    def __getbbox_1dim__(distance, distype, minRefer, maxRefer, WoH):
        boxfeat1dim = torch.zeros([2], dtype=torch.float32)
        maxtype, maxtypeR = 0, distype[0]
        for i in range(1, len(distype)):
            if distype[i] > maxtypeR:
                maxtype, maxtypeR = i, distype[i]
        minOrmaxRefer = maxtype // 2
        referIndex = maxtype % 2
        referArr = [minRefer , maxRefer]
        if minOrmaxRefer == 0:
            boxfeat1dim[0] = referArr[referIndex] - distance
            boxfeat1dim[1] = referArr[referIndex] - distance + WoH
        else:
            boxfeat1dim[1] = referArr[referIndex] - distance
            boxfeat1dim[0] = referArr[referIndex] - distance - WoH
        
        return boxfeat1dim


    @staticmethod
    def calculateEdgeDistance(targetEdge, edge1, edge2):
        #returns distance and type
        distance, type = 0, 0
        if targetEdge < edge1:
            distance, type = edge1 - targetEdge, 0
        if targetEdge >= edge1 and targetEdge <= edge2 :
            if targetEdge - edge1 <= edge2 - targetEdge:
                distance, type = targetEdge - edge1, 1
            else:
                distance, type = edge2 - targetEdge, 2
        if targetEdge > edge2:
            distance, type = targetEdge -  edge2, 3

        return distance, type
    

    # @staticmethod
    # def calculateBBoxThroughDetailedPosition(detailedRP, bboxChild1, w, h):
    #     newBBox = torch.zeros([4], dtype=torch.float32)
    #     relativeType = PositionMapper.getEgdeRelativeType(detailedRP[4:8])
    #     edgeX, type = PositionMapper.calculateEdgeDistanceReverseLtoR(detailedRP[0], relativeType, bboxChild1[0], bboxChild1[2])
    #     newBBox[0] = edgeX
    #     newBBox[2] = edgeX + w

    #     # Y dimention
    #     relativeType = PositionMapper.getEgdeRelativeType(detailedRP[12:16])
    #     edgeY, type = PositionMapper.calculateEdgeDistanceReverseLtoR(detailedRP[2], relativeType, bboxChild1[1], bboxChild1[3])
    #     newBBox[1] = edgeY
    #     newBBox[3] = edgeY + h
    #     return newBBox
    @staticmethod
    def calculateBBoxThroughDetailedPosition(rp, detailedRP, bboxChild1, w, h):
        type = PositionMapper.onehotToPosition(rp)
        newBBox = torch.zeros([4], dtype=torch.float32)
        rminx, rminy, rmaxx, rmaxy = bboxChild1[0], bboxChild1[1], bboxChild1[2], bboxChild1[3]
        dx, dy, dx1, dx2 = detailedRP[0], detailedRP[1], detailedRP[2], detailedRP[3]
        if type == "left":
            newBBox[0], newBBox[1], newBBox[2], newBBox[3] = \
                rminx - dx - w, rmaxy - dy, rminx - dx, rmaxy - dy + h
        if type == "enclosed":
            newBBox[0], newBBox[1], newBBox[2], newBBox[3] = \
                rminx - dx, rminy - dy, rminx - dx + w, rminy - dy + h
        if type == "right":
            newBBox[0], newBBox[1], newBBox[2], newBBox[3] = \
                rmaxx - dx, rmaxy - dy, rmaxx - dx + w, rmaxy - dy + h
        if type == "bottomLeft":
            newBBox[0], newBBox[1], newBBox[2], newBBox[3] = \
                rminx - dx, rmaxy - dy, rminx - dx + w, rmaxy - dy + h
        if type == "bottom":
            newBBox[0], newBBox[1], newBBox[2], newBBox[3] = \
                rminx - dx, rmaxy - dy, rminx - dx + w, rmaxy - dy + h 
        if type == "bottomRight":
            newBBox[0], newBBox[1], newBBox[2], newBBox[3] = \
                rminx - dx - w, rmaxy - dy, rminx - dx, rmaxy - dy + h
        if type == "wideBottom":
            newBBox[0], newBBox[1], newBBox[2], newBBox[3] = \
                rminx - dx1, rmaxy - dy, rmaxx - dx2, rmaxy - dy + h
        return newBBox
        
        
    @staticmethod
    def getEgdeRelativeType(onehot):
        max, index = -0x0FFFFFFF, 0
        for i in range(0, onehot.size()[-1]):
            if onehot[i] > max:
                max, index = onehot[i], i
        return index

    # known left calulate right
    @staticmethod
    def calculateEdgeDistanceReverseLtoR(distance, typeTensor, edge1, edge2):
        if typeTensor == 0:
            return edge1 - distance, 1
        if typeTensor == 1:
            return edge1 + distance, 1
        if typeTensor == 2:
            return edge2 - distance, 2
        if typeTensor == 3:
            return edge2 + distance, 2

    # known right calulate left
    @staticmethod
    def calculateEdgeDistanceReverseRtoL(distance, typeTensor, edge1, edge2):
        if typeTensor == 0:
            return edge1 + distance, 1
        if typeTensor == 1:
            return edge1 - distance, 1
        if typeTensor == 2:
            return edge2 + distance, 2
        if typeTensor == 3:
            return edge2 - distance, 2


    @staticmethod
    def merge2BBox(bbox1, bbox2):
        minx, miny = min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1])
        maxx, maxy = max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])
        newbox = torch.tensor([minx, miny, maxx, maxy], dtype=torch.float32)
        return newbox
    
    @staticmethod
    def applyOffset(targetBBox, offset):
        scalarx = 1
        scalary = 1
        if offset[2] > 1:
            scalarx = 1 / offset[2]
        if offset[3] > 1:
            scalary = 1 / offset[3]
        targetBBox[0], targetBBox[2] = (targetBBox[0] - offset[0]) * scalarx, (targetBBox[2] - offset[0]) * scalarx
        targetBBox[1], targetBBox[3] = (targetBBox[1] - offset[1]) * scalary, (targetBBox[3] - offset[1]) * scalary
        return targetBBox
