import math
import logging


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    Precision = []
    Recall = []
    F1 = []
    NDCG = []
    One_Call = []
    MRR = []

    # Out the recommendation
    # for i in range(len(predictedIndices)):
    #     if len(GroundTruth[i]) != 0:
    #         rec_list = []
    #         for j in range(100):
    #             rec_list.append(predictedIndices[i][j])
    #         logging.info('%d: %s'%(i, rec_list))

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForF1 = 0
        sumForNDCG = 0
        sumForOne_Call = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0 / (j + 1.0))
                            mrrFlag = False
                        userHit += 1

                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1

                if (idcg != 0):
                    ndcg += (dcg / idcg)

                precision_u = userHit / topN[index]
                recall_u = userHit / len(GroundTruth[i])

                sumForPrecision += precision_u
                sumForRecall += recall_u
                if (precision_u + recall_u) != 0:
                    sumForF1 += 2 * (precision_u * recall_u / (precision_u + recall_u))
                sumForNDCG += ndcg
                sumForOne_Call += 1 if userHit > 0 else 0
                sumForMRR += userMRR

        Precision.append(sumForPrecision)
        Recall.append(sumForRecall)
        F1.append(sumForF1)
        NDCG.append(sumForNDCG)
        One_Call.append(sumForOne_Call)
        MRR.append(sumForMRR)

        # Precision.append(sumForPrecision / test_userNum)
        # Recall.append(sumForRecall / test_userNum)
        # NDCG.append(sumForNDCG / test_userNum)
        # MRR.append(sumForMRR / test_userNum)

    return Precision, Recall, F1, NDCG, One_Call, MRR
