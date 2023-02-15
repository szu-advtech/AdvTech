import math
import sys
import time
import torch
import transforms
import numpy as np

def evaluate(model, data_loader, device):
    model.eval()

    F1list = []
    RTlist = []
    RFlist = []
    
    F1list1 = []
    RTlist1 = []
    RFlist1 = []

    for step, [image, pore,minutiae] in enumerate(data_loader):

        output = model(image.to(device))
        output = torch.squeeze(output)
        pre_pore_coords = find_coords(output[0])
        pre_minutiae_coords = find_coords(output[1])
        GT_pore_coords = pore.numpy()
        GT_pore_coords = np.squeeze(GT_pore_coords)
        GT_minutiae_coords = minutiae.numpy()
        GT_minutiae_coords = np.squeeze(GT_minutiae_coords)
        TP, FN, FP = eval(pre_pore_coords,GT_pore_coords, 5)
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2*precision*recall/(precision + recall)
        RT = recall
        RF = 1 - precision
        
        F1list.append(F1)
        RTlist.append(RT)
        RFlist.append(RF)
        
        TP1, FN1, FP1 = eval(pre_minutiae_coords,GT_minutiae_coords, 12)
        precision1 = TP1 / (TP1 + FP1) if (TP1 + FP1) != 0 else 0
        recall1 = TP1 / (TP1 + FN1)  if (TP1 + FN1) != 0 else 0
        F11 = 2*precision1*recall1/(precision1 + recall1) if (precision1 + recall1) != 0 else 0
        RT1 = recall1
        RF1 = 1 - precision1
        
        F1list1.append(F11)
        RTlist1.append(RT1)
        RFlist1.append(RF1)
    return np.mean(F1list), np.mean(RTlist), np.mean(RFlist), np.mean(F1list1), np.mean(RTlist1), np.mean(RFlist1)




def find_coords(matrix):
    radius = 1
    coords = []
    H, W = matrix.size()
    for i in range(H):
        for j in range(W):
            if matrix[i][j] < 0.25:
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
        GT_point = GT[i]    # 取出第i个groundTruth Point
        for j in range(len(predicted)):
            # pred_x = predicted[j][0]
            # pred_y = predicted[j][1]
            pred_point = predicted[j]    # 取出第j个predicted Point
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

