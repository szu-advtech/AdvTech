import numpy as np
import math
from scipy.optimize import linear_sum_assignment

def weightBetTwoBlocks(bbox1, type1, bbox2, type2):
    if (type1 != type2):
        return 0

    #α(B1, B2) = min(w1h1, w2h2)^(1/2)
    alphaB1B2 = math.sqrt( min(bbox1[2] * bbox1[3], bbox2[2] * bbox2[3]) )

    # ∆C(B1,B2)
    center1 = (bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2)
    center2 = (bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2)
    deltaCB1B2 = math.sqrt(math.pow(center1[0] - center2[0], 2) + math.pow(center1[1] - center2[1], 2))
    Cs = 2

    # ∆S(B1,B2)
    deltaSB1B2 = abs(bbox1[2] - bbox2[2]) + abs(bbox1[3] - bbox2[3])
    
    weight = alphaB1B2 * math.pow(2, -deltaCB1B2-Cs*deltaSB1B2)
    return weight

#输入 layout1的bbox-type列表， latout2的bbox-type列表， bbox和对应type的Index应该相同
def calculateDocSim(bboxList1:list, typeList1:list, bboxList2:list, typeList2:list):
    if len(bboxList1) != len(typeList1) or len(bboxList2) != len(typeList2):
        print("[DocSim]计算docSim时输入数据错误")
    
    num1 = len(bboxList1)
    num2 = len(bboxList2)
    # Ensure that num1 >= num2, if num2 > num1, then exchange 1 and 2
    if num2 > num1:
        temp = num2
        num2 = num1
        num1 = temp
        temp = bboxList2
        bboxList2 = bboxList1
        bboxList1 = temp
        temp = typeList2
        typeList2 = typeList1
        typeList1 = temp
    # -----
    # Then calculate metric matrix
    mmatrix = np.zeros((num2, num1), dtype=float)
    for i in range(0, num2):
        for j in range(0, num1):
            mmatrix[i][j] = - weightBetTwoBlocks(bboxList1[j], typeList1[j], bboxList2[i], typeList2[i])
    row_ind, col_ind = linear_sum_assignment(mmatrix)

    # print(mmatrix)
    return -mmatrix[row_ind, col_ind].sum()

# print(calculateDocSim([[0.5, 0.5, 0.3, 0.3], [0, 0.5, 0.3, 0.5], [0, 0, 0.6, 0.6]], ["text", "text", "text"], \
#                       [[0.5, 0.5, 0.3, 0.3], [0, 0.5, 0.3, 0.5], [0, 0, 0.6, 0.6]], ["text", "text", "text"]) )

