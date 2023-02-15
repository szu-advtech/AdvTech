import os
import random
import time
import numpy as np
import torch
import xlwt
import dataset_metatest
import datapre
import metatrain
import TODO_save_data
from simple_cnaps_src.simple_cnaps_l1 import SimpleCnaps
from path_index import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ways=32
seed=42
book = xlwt.Workbook(encoding='utf-8',style_compression=0)
sheet = book.add_sheet("seed_"+str(seed)+"_way_"+str(train_ways),cell_overwrite_ok=True)
sheet.write(0,0,"testway/testshot")

#NEEDSHOT=[1,5]
NEEDSHOT=[metatrain_testways]
NEEDWAY=[5]
for needshot in NEEDSHOT:
    sheet.write(0, needshot, "testshot_"+str(needshot))#first row second line
for needway in NEEDWAY:
    sheet.write(int((needway-6)/2+1), 0, "testway_"+str(needway))  # first row second line

for needshot in NEEDSHOT:
    for needway in NEEDWAY:
        shot=needshot
        test_samples = shot#

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = SimpleCnaps(device=device, use_two_gpus=False).to(device)
        state_dict = torch.load(metatest_testmodel)
        model.load_state_dict(state_dict)
        test_dataset = datapre.MetaDataset(dataset_metatest.digits_set(data1=TODO_save_data.get_data(needway, seed=42)))  # total 26 letters
        transforms = \
        [
            datapre.transforms.NTask(test_dataset),
            datapre.transforms.NWays(test_dataset, needway),  # testways
            datapre.transforms.KShots(test_dataset, test_samples),
            datapre.transforms.LoadData(test_dataset),
        ]
        test_dataset_ = datapre.TaskDataset(dataset=test_dataset, task_transforms=transforms)
        print("train_ways:"+str(train_ways)+"_"+"seed:"+str(seed)+"_"+"test_ways:"+str(needway)+"_"+"testshots:"+str(test_samples))

        model.eval()
        with torch.no_grad():
            acc, time = metatrain.cnaps_test(test_dataset_, model, device)
        sheet.write(int((needway-6)/2+1),needshot,acc)
    book.save("testshot.xls")
