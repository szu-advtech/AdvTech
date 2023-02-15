#!/usr/bin/env python
# encoding: utf-8
# author: Hu Xin
# email: qzlyhx@gmail.com
# time: 2022.8.15
# description:
#   This file contructs a dataset that is compatible with pytorch dataset using Document Loader
from torch.utils.data import DataLoader, Dataset
import os

from utils.documentLoader import Document
from utils.layoutConverter import generateDocumentFromJsonData

def collate_fn(doc):
    return doc

# 50 imgs as test set and others training
class DocumentTrainDataset(Dataset):
    def __init__(self, root_dir, imageFolder="Images/", xmlFolder="XML/") -> None:
        self.xmlPath = root_dir + xmlFolder
        self.imagePath = root_dir + imageFolder
        # load all file names
        self.xmlFileNames = []
        self.imageFileNames = []
        for path in os.listdir(self.xmlPath):
            # check if current path is a file
            if os.path.isfile(os.path.join(self.xmlPath, path)):
                self.xmlFileNames.append(path)
                self.imageFileNames.append(path.split('-')[-1].split('.')[-2] + ".tif")
        self.xmlFileNames = self.xmlFileNames[0:len(self.xmlFileNames)]
        self.imageFileNames = self.imageFileNames[0:len(self.imageFileNames)]
        #contruct Documents
        self.documentSet = []

        for i in range(0, len(self.imageFileNames)):
            self.documentSet.append(Document(imagePath=self.imagePath + self.imageFileNames[i], labelPath=self.xmlPath + self.xmlFileNames[i]))
    
    def __len__(self):
        return len(self.imageFileNames)
    
    def __getitem__(self, idx):
        return self.documentSet[idx]

# 50 imgs for test set
class DocumentTestDataset(Dataset):
    def __init__(self, root_dir, imageFolder="Images/", xmlFolder="XML/") -> None:
        self.xmlPath = root_dir + xmlFolder
        self.imagePath = root_dir + imageFolder
        # load all file names
        self.xmlFileNames = []
        self.imageFileNames = []
        for path in os.listdir(self.xmlPath):
            # check if current path is a file
            if os.path.isfile(os.path.join(self.xmlPath, path)):
                self.xmlFileNames.append(path)
                self.imageFileNames.append(path.split('-')[-1].split('.')[-2] + ".tif")
        self.xmlFileNames = self.xmlFileNames[0:50]
        self.imageFileNames = self.imageFileNames[0:50]
        #contruct Documents
        self.documentSet = []
        for i in range(0, len(self.imageFileNames)):
            self.documentSet.append(Document(imagePath=self.imagePath + self.imageFileNames[i], labelPath=self.xmlPath + self.xmlFileNames[i]))
    
    def __len__(self):
        return len(self.imageFileNames)
    
    def __getitem__(self, idx):
        return self.documentSet[idx]



class LayoutTrainDataset(Dataset):
    def __init__(self, root_dir) -> None:
        # load all file names
        self.root_dir = root_dir
        self.jsonFileNames = []

        with open(root_dir + "train.txt", "r") as f:
            try:
                while True:
                    linedata = f.readline()
                    linedata = linedata[0:(len(linedata) - 1)]
                    if linedata:
                        self.jsonFileNames.append(linedata + ".json")
                    else:
                        break;
            finally:
                f.close()
        self.documentSet = []

        for i in range(0, len(self.jsonFileNames)):
            # print("--------------------------------------------\n\n\n")
            # print(self.jsonFileNames[i])
            doc = generateDocumentFromJsonData(path=root_dir + self.jsonFileNames[i])
            self.documentSet.append(doc)
    
    def __len__(self):
        return len(self.jsonFileNames)
    
    def __getitem__(self, idx):
        return self.documentSet[idx]

# 1000 imgs as test set and others training
class LayoutTestDataset(Dataset):
    def __init__(self, root_dir) -> None:
        # load all file names
        self.root_dir = root_dir
        self.jsonFileNames = []
        with open(root_dir + "test.txt", "r") as f:
            try:
                while True:
                    linedata = f.readline()
                    linedata = linedata[0:(len(linedata) - 1)]
                    if linedata:
                        self.jsonFileNames.append(linedata + ".json")
                    else:
                        break;
            finally:
                f.close()
        #contruct Documents
        self.documentSet = []

        for i in range(0, len(self.jsonFileNames)):
            doc = generateDocumentFromJsonData(path=root_dir + self.jsonFileNames[i])
            self.documentSet.append(doc)
    
    def __len__(self):
        return len(self.jsonFileNames)
        
    def __getitem__(self, idx):
        return self.documentSet[idx]