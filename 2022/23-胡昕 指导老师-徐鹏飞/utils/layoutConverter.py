import json

from models.tree import *
from utils.documentLoader import *
from utils.positionMapper import *

labelmap = {
    'vertical_branch':"branch",
    'horizontal_branch':"branch",
    'image':"imageRegion",
    'text':"paragraph",
    'none':"floating",
    'title':"heading",
    'list':"list",
    'table':"table",
    'figure':"imageRegion",

}
# get sort key of region

def __regionSortByY(region):
    return region['abs_box'][1] # minY


def __regionSortByX(region):
    return region['abs_box'][0] # minX

def generateDocumentFromJsonData(path):
    # read json data
    with open(path, "r") as f:
            jsData = json.load(f)
    layoutData = jsData
    # print(path)
    # print(len(layoutData['children'][1]['children'][2]['children']))
    # convert layout data to Document class
    # only need to generate a layout tree and set it to a Document class

    # step1: get all regions from json file
    regions = []
    __get_regions(layoutData, regions)
    
    # step2: reorder regions
    regions.sort(key=__regionSortByX)
    regions.sort(key=__regionSortByY)

    # step3: Generate Region type class
    lregions =  len(regions)
    for i in range(0, lregions):
        reg = Region(type=labelmap[regions[i]['label']])
        bbox = regions[i]['abs_box']
        reg.coordinates = [
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            (bbox[0], bbox[1] + bbox[3]),
        ]
        reg.generateAtomicUnitFeature(1, 1)
        regions[i] = reg

    # step4: Generate layout tree
    targetRegions = regions
    if len(regions) == 0:
        targetRegions.append()
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
    layoutTree = Tree(rootNode=rootNow)

    # step5: Generate Document
    newDoc = Document()
    newDoc.documentLabelPath = path.split("/")[-1].split(".")[0]
    newDoc.regions = targetRegions
    newDoc.layoutTree = layoutTree
    return newDoc


def __get_regions(layoutData, regions):
    
    if layoutData['label'] == "vertical_branch" or layoutData['label'] == "horizontal_branch":
        if "children" in layoutData.keys():
            for child in layoutData['children']:
                __get_regions(child, regions)
        else:
            print("Branch but no children, revise it to paragraph")
            layoutData['label'] = "paragraph"
    else:
        if layoutData['label'] != "none":
            regions.append(layoutData)


