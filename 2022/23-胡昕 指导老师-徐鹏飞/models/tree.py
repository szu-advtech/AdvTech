
# this structure repersent the structure of a tree Node

from utils.config import getDevice


class Node():
    def __init__(self, nodeData = None, child1 = None, child2 = None, leafAtomicUnitData = None, nodeRelativePositionData=None, nodeDetailedRelativePositionData = None, \
                       bbox=None) -> None:
        # set two childs
        self.child1 = child1
        self.child2 = child2
        self.nodeData = nodeData
        self.leafAtomicUnitData = leafAtomicUnitData
        self.nodeRelativePositionData = nodeRelativePositionData
        self.nodeDetailedRelativePositionData = nodeDetailedRelativePositionData
        self.nodeNotLeafProbability = None
        self.bbox = bbox
        # get device then put these tensors to the corressponding devices
        device = getDevice()
        if self.nodeData != None:
            self.nodeData = self.nodeData.to(device)
        if self.leafAtomicUnitData != None:
            self.leafAtomicUnitData = self.leafAtomicUnitData.to(device)
        if self.nodeRelativePositionData != None:
            self.nodeRelativePositionData = self.nodeRelativePositionData.to(device)
        if self.nodeNotLeafProbability != None:
            self.nodeNotLeafProbability = self.nodeNotLeafProbability.to(device)
        if self.nodeDetailedRelativePositionData != None:
            self.nodeDetailedRelativePositionData = self.nodeDetailedRelativePositionData.to(device)

    def setChild1(self, child1):
        self.child1 = child1

    def setChild2(self, child2):
        self.child2 = child2

    def setNodeData(self, data):
        device = getDevice()
        self.nodeData = data.to(device)
        
    def setLeafAtomicUnitData(self, data):
        device = getDevice()
        self.leafAtomicUnitData = data.to(device)
    
    def setNodeRelativePositionData(self, data):
        device = getDevice()
        self.nodeRelativePositionData = data.to(device)
    
    def setNodeDetailedRelativePositionData(self, data):
        device = getDevice()
        self.nodeDetailedRelativePositionData = data.to(device)



class Tree():
    def __init__(self, rootNode = None) -> None:
        # set Root Node
        self.rootNode = rootNode
        
        # other information that may be useds

class TreeStructureAnalyser():
    @staticmethod
    def calculateDepthWidth(TreeNode:Node) -> tuple:
        if TreeNode == None:
            return 0, 0
        height, width, loopNodeNum = 0, 0, 1
        # Use BFS to calculate
        nodeList = list()
        nodeList.append(TreeNode)
        while loopNodeNum > 0:
            height = height + 1
            if loopNodeNum > width:
                width = loopNodeNum
            nextLoopNum = 0
            for i in range(0, loopNodeNum):
                tnode = nodeList.pop(0)
                if tnode.child1 != None:
                    nodeList.append(tnode.child1)
                    nextLoopNum = nextLoopNum + 1
                if tnode.child2 != None:
                    nodeList.append(tnode.child2)
                    nextLoopNum = nextLoopNum + 1
            # reset loop variables
            loopNodeNum = nextLoopNum
        
        return height, width



