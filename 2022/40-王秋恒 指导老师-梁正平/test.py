
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt


from torchvision.transforms import functional as F
import PIL.Image as Image

def predict(fold):
    
    eval_list = [range(25,31),range(19,25), range(13,19),range(7,13),range(1,7)]
    result_file = "./minu_acu2.txt"
    radius = 1
    with open(result_file, "a") as f:
            
            txt = "fold" + str(fold+1)
            f.write(txt + "\n")
    for k in range(1, 16):
        F1list = []
        RTlist = []
        RFlist = []
        for i in range(1,31):
            img_bi_pth = "./predict_5_fold/" + f"{fold+1}" + "/pth" + f"{k}" + "/" + f"{i}" + "minutiae.bmp"
            img_bi = Image.open(img_bi_pth)
            img_bi = np.array(img_bi)
            pre_bi = find_coords(img_bi, radius)
            
            GT_bi_pth  = "./minutiae_coord_v1/" + f"{i}" + ".txt"
            GT_bi = []
            with open(GT_bi_pth) as f:
                for line in f:
                    tmp = line.split("\t")
                    GT_bi.append((int(tmp[0]), int(tmp[1])))

            TP, FN, FP = eval(pre_bi, GT_bi, 16)
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN)  if (TP + FP) != 0 else 0
            F1 = 2* precision * recall / (precision + recall) if (precision + recall) != 0 else 0
            RT = recall
            RF = 1 - precision
            if i in eval_list[fold]:
                F1list.append(F1)
                RTlist.append(RT)
                RFlist.append(RF)

            print("----------img", i, "--------------")
            print(TP, FN, FP)
            print(RT, RF)

        F1mean = np.mean(F1list)
        RTmean = np.mean(RTlist)
        RFmean = np.mean(RFlist)

        print("F1:", F1mean, "RT:", RTmean, "RF:", RFmean)
        with open(result_file, "a") as f:
            save_info = [f"{F1mean:.4f}", f"{RTmean:.4f}", f"{RFmean:.4f}"]
            txt = "epoch:{} {}".format(k, '  '.join(save_info))
            f.write(txt + "\n")


def find_coords(matrix, radius):
    
    coords = []
    H, W = 240, 320
    for i in range(H):
        for j in range(W):
            if matrix[i][j] < 50:
                matrix[i][j] = 0
    for i in range(H):
        for j in range(W):
            # 设置一个flag
            flag = 0
            # 取出一个窗口
            up = i + radius if i + radius < H - 1 else H - 1
            button = i - radius if i - radius > 0 else 0
            left = j - radius if j - radius > 0 else 0
            right = j + radius if j + radius < W - 1 else W - 1
            # 遍历窗口中的其他值
            for x in range(button, up + 1):
                for y in range(left, right + 1):
                    if (x, y) != (i, j) and matrix[x][y] >= matrix[i][j]:
                        flag = 1
            if flag == 0:
                coords.append((i, j))
    return coords


def eval(predicted, GT, threshold):
    TP, FP, FN = 0, 0, 0
    for i in range(len(GT)):
        # GT_x = GT[i][0] # point_x
        # GT_y = GT[i][1] # point_y
        flag1 = 0
        GT_point = GT[i]
        for j in range(len(predicted)):
            # pred_x = predicted[j][0]
            # pred_y = predicted[j][1]
            pred_point = predicted[j]
            distance = np.sqrt(np.sum((np.array(pred_point) - np.array(GT_point)) ** 2))
            if distance <= threshold:  # find the point
                TP += 1  # True Positive (pred = 1 & label = 1)
                flag1 = 1
                break
        if flag1 == 0:  # miss the point
            FN += 1  # False Negative (pred = 0 & label = 1)

    for j in range(len(predicted)):
        # pred_x = predicted[j][0]
        # pred_y = predicted[j][1]
        flag2 = 0
        pred_point = predicted[j]
        for i in range(len(GT)):
            # GT_x = GT[i][0] # center_w
            # GT_y = GT[i][1] # center_h
            GT_point = GT[i]
            distance = np.sqrt(np.sum((np.array(pred_point) - np.array(GT_point)) ** 2))
            if distance <= threshold:  # find the minutiae
                flag2 = 1
                break
        if flag2 == 0:  # spurious minutiae
            FP += 1  # False Positivate (pred = 1 & label = 0)

    return TP, FN, FP

for i in range(5):
    predict(i)
